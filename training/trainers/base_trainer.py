import os
import glob
import torch
import traceback
from admin import loading, multigpu


class BaseTrainer:
    """Base trainer class. Contains functions for training and saving/loading checkpoints.
    Trainer classes should inherit from this one and overload the train_epoch function."""

    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None, make_initial_validation=False):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                      epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
            make_initial_validation - bool, make initial validation before first training epoch?
        """
        self.actor = actor
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loaders = loaders
        self.make_initial_validation = make_initial_validation
        # if we want to first evaluate on validation dataset, after loading the weights for instance

        self.update_settings(settings)

        self.just_started = False  # to sample new dataset items only at the next epoch
        self.epoch = 0
        self.stats = {}
        self.best_val = float("Inf")  # absolute best val, to know when to save the checkpoint.
        self.epoch_of_best_val = 0
        self.current_best_val = None  # will be updated at each epoch, if we do validation

        self.device = getattr(settings, 'device', None)
        if self.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() and settings.use_gpu else "cpu")

        self.actor.to(self.device)  # puts the network to GPU

    def update_settings(self, settings=None):
        """Updates the trainer settings. Must be called to update internal settings."""
        if settings is not None:
            self.settings = settings

        if self.settings.env.workspace_dir is not None:
            self.settings.env.workspace_dir = os.path.expanduser(self.settings.env.workspace_dir)
            # self._base_save_dir = os.path.join(self.settings.env.workspace_dir, 'checkpoints')
            # use the same base directory for checkpoints and tensorboard, and home is env.workspace_dir
            self._base_save_dir = self.settings.env.workspace_dir
            if not os.path.exists(self._base_save_dir):
                os.makedirs(self._base_save_dir)
        else:
            self._base_save_dir = None

    def train(self, max_epochs, load_latest=False, fail_safe=True, load_ignore_fields=None):
        """Do training for the given number of epochs.
        args:
            max_epochs - Max number of training epochs,
            load_latest - Bool indicating whether to resume from latest epoch.
            fail_safe - Bool indicating whether the training to automatically restart in case of any crashes.
        """

        self.just_started = True
        epoch = -1
        num_tries = 2
        for i in range(num_tries):
            try:
                if load_latest:
                    self.load_checkpoint(additional_ignore_fields=load_ignore_fields)

                for epoch in range(self.epoch+1, max_epochs+1):
                    self.epoch = epoch

                    # do one training epoch
                    self.train_epoch()

                    # update scheduler
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                    # save best checkpoint
                    if self.current_best_val is not None and self.current_best_val < self.best_val:
                        print('VALIDATION IMPROVED ! From best value = {} at epoch {} to '
                              'best value = {} at current epoch {}'.
                              format(self.best_val, self.epoch_of_best_val, self.current_best_val, self.epoch))
                        self.best_val = self.current_best_val
                        self.epoch_of_best_val = self.epoch

                        self.save_checkpoint(name='model_best')

                    self.just_started = False  # to enable resampling of dataset item at the next epoch
                    # save checkpoint
                    if self._base_save_dir:
                        self.save_checkpoint()
                        self.delete_old_checkpoints()  # keep only the most recent set of checkpoints

            except:
                print('Training crashed at epoch {}'.format(epoch))
                if fail_safe:
                    self.epoch -= 1
                    load_latest = True
                    print('Traceback for the error!')
                    print(traceback.format_exc())
                    print('Restarting training from last epoch ...')
                else:
                    raise

        print('Finished training!')

    def train_epoch(self):
        raise NotImplementedError

    def delete_old_checkpoints(self):
        """Delete all but the num_keep last saved checkpoints."""
        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        net_type = type(net).__name__

        ckpts = sorted(glob.glob('{}/{}/{}_ep*.pth.tar'.format(self._base_save_dir, self.settings.project_path,
                                                               net_type)))
        ckpts_to_remove = sorted(ckpts)[:-self.settings.keep_last_checkpoints]
        if len(ckpts_to_remove) > 0:
            for ckpt in ckpts_to_remove:
                os.remove(ckpt)

    def save_checkpoint(self, name=None):
        """Saves a checkpoint of the network and other variables."""

        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        actor_type = type(self.actor).__name__
        net_type = type(net).__name__
        state = {
            'epoch': self.epoch,
            'actor_type': actor_type,
            'net_type': net_type,
            'state_dict': net.state_dict(),
            'net_info': getattr(net, 'info', None),
            'constructor': getattr(net, 'constructor', None),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'stats': self.stats,
            'best_val': self.best_val,
            'epoch_of_best_val': self.epoch_of_best_val
        }

        directory = '{}/{}'.format(self._base_save_dir, self.settings.project_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        if name is not None:
            # give a specific name to the file, such as model_best
            # First save as a tmp file
            tmp_file_path = '{}/{}_{}.tmp'.format(directory, net_type, name)
            torch.save(state, tmp_file_path)

            file_path = '{}/{}_{}.pth.tar'.format(directory, net_type, name)
        else:
            # First save as a tmp file
            tmp_file_path = '{}/{}_ep{:04d}.tmp'.format(directory, net_type, self.epoch)
            torch.save(state, tmp_file_path)

            file_path = '{}/{}_ep{:04d}.pth.tar'.format(directory, net_type, self.epoch)

        # Now rename to actual checkpoint. os.rename seems to be atomic if files are on same filesystem. Not 100% sure
        os.rename(tmp_file_path, file_path)

    def load_checkpoint(self, checkpoint=None, fields=None, ignore_fields=None, additional_ignore_fields=None,
                        load_constructor=False):
        """Loads a network checkpoint file.
        Can be called in three different ways:
            load_checkpoint():
                Loads the latest epoch from the workspace. Use this to continue training.
            load_checkpoint(epoch_num):
                Loads the network at the given epoch number (int).
            load_checkpoint(path_to_checkpoint):
                Loads the file from the given absolute path (str).
        """

        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        actor_type = type(self.actor).__name__
        net_type = type(net).__name__

        resume = True
        if checkpoint is None:
            # Load most recent checkpoint
            checkpoint_list = sorted(glob.glob('{}/{}/{}_ep*.pth.tar'.format(self._base_save_dir,
                                                                             self.settings.project_path, net_type)))
            if checkpoint_list:
                checkpoint_path = checkpoint_list[-1]
            else:
                print('No matching checkpoint file found')
                return
        elif isinstance(checkpoint, int):
            # Checkpoint is the epoch number
            checkpoint_path = '{}/{}/{}_ep{:04d}.pth.tar'.format(self._base_save_dir, self.settings.project_path,
                                                                 net_type, checkpoint)
        elif isinstance(checkpoint, str):
            # checkpoint is the path
            if os.path.isdir(checkpoint):
                checkpoint_list = sorted(glob.glob('{}/*_ep*.pth.tar'.format(checkpoint)))
                if checkpoint_list:
                    checkpoint_path = checkpoint_list[-1]
                else:
                    raise Exception('No checkpoint found')
            else:
                checkpoint_path = os.path.expanduser(checkpoint)
                resume = False

        else:
            raise TypeError

        # Load network
        print('current epoch: ',self.epoch)
        print('load checkpoint from ', checkpoint_path)
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
        checkpoint_net_type = checkpoint_dict['net_type']

        assert net_type == checkpoint_net_type or checkpoint_net_type in net_type, \
            f'Network is not of correct type: Current --> {net_type} ; Checkpoint --> {checkpoint_net_type}'

        if fields is None:
            fields = checkpoint_dict.keys()
        if ignore_fields is None:
            ignore_fields = ['settings']

        # Never load the scheduler. It exists in older checkpoints.
        ignore_fields.extend(['lr_scheduler', 'constructor', 'net_type', 'actor_type', 'net_info'])
        if additional_ignore_fields is not None:
            ignore_fields.extend(additional_ignore_fields)

        # Load all fields
        for key in fields:
            if key in ignore_fields:
                continue
            if key == 'state_dict':
                net.load_state_dict(checkpoint_dict[key])
            elif key == 'optimizer':
                if checkpoint_dict[key]:
                    self.optimizer.load_state_dict(checkpoint_dict[key])
                '''
                # now individually transfer the optimizer parts...
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)'''
            else:
                setattr(self, key, checkpoint_dict[key])

        if not resume:
            self.epoch = 0
            self.stats = {}
            self.best_val = float("Inf")  # absolute best val, to know when to save the checkpoint.
            self.epoch_of_best_val = 0

        print('current epoch: ', self.epoch)
        # Set the net info
        if load_constructor and 'constructor' in checkpoint_dict and checkpoint_dict['constructor'] is not None:
            net.constructor = checkpoint_dict['constructor']
        if 'net_info' in checkpoint_dict and checkpoint_dict['net_info'] is not None:
            net.info = checkpoint_dict['net_info']

        # Update the epoch in lr scheduler
        if 'epoch' in fields:
            # here we do not save and update the scheduler to make it easier to change the lr
            self.lr_scheduler.last_epoch = self.epoch

        return True
