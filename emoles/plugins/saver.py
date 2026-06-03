from emoles.plugins.base_plugin import Plugin
from collections import defaultdict
import logging
import os
import time
import torch

log = logging.getLogger(__name__)

class Saver(Plugin):
    def __init__(self, interval=None):
        if interval is None:
            interval = [(1, 'iteration'), (1, 'epoch')]
        super(Saver, self).__init__(interval)
        self.best_loss = 1e7
        self.best_quene = []
        self.latest_quene = []

    def register(self, trainer, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.trainer = trainer
        self.push = False
        

    def iteration(self, **kwargs):
        suffix = ".iter{}".format(self.trainer.iter)
        max_ckpt = self.trainer.train_options["max_ckpt"]

        name = self.trainer.model.name+suffix
        self.latest_quene.append(name)
        
        if len(self.latest_quene) > max_ckpt:
            delete_name = self.latest_quene.pop(0)
            delete_path = os.path.join(self.checkpoint_path, delete_name+".pth")
            try:        
                os.remove(delete_path)
            except:
                log.info(f"Failed to delete the checkpoint file {delete_path}.")
                
        self._save(
            name=name,
            model=self.trainer.model,
            model_options=self.trainer.model.model_options,
            common_options=self.trainer.common_options,
            train_options=self.trainer.train_options,
            )
        
        latest_symlink = os.path.join(self.checkpoint_path, self.trainer.model.name + ".latest.pth")
        if os.path.lexists(latest_symlink):
            os.unlink(latest_symlink)
        latest_ckpt = os.path.join(self.checkpoint_path, name+".pth")
        latest_ckpt_abs_path = os.path.abspath(latest_ckpt)
        if not os.path.exists(latest_ckpt_abs_path):
            raise FileNotFoundError(f"Source file {latest_ckpt_abs_path} does not exist.")
        os.symlink(latest_ckpt_abs_path, latest_symlink)

    def epoch(self, **kwargs):

        updated_loss = self.trainer.stats.get('validation_loss')
        if updated_loss is not None:
            updated_loss = updated_loss.get('epoch_mean',1e6)
        else:
            updated_loss = self.trainer.stats.get("train_loss").get("epoch_mean",1e6)

        max_ckpt = self.trainer.train_options["max_ckpt"]

        if updated_loss < self.best_loss:
            suffix = ".ep{}".format(self.trainer.ep)
            name = self.trainer.model.name+suffix
            self.best_quene.append(name)
            if len(self.best_quene) > max_ckpt:
                delete_name = self.best_quene.pop(0)
                delete_path = os.path.join(self.checkpoint_path, delete_name+".pth")
                os.remove(delete_path)

            self._save(
                name=name,
                model=self.trainer.model,
                model_options=self.trainer.model.model_options,
                common_options=self.trainer.common_options,
                train_options=self.trainer.train_options,
                )
            
            self.best_loss = updated_loss

            best_symlink = os.path.join(self.checkpoint_path, self.trainer.model.name + ".best.pth")
            if os.path.lexists(best_symlink):
                os.unlink(best_symlink)
            best_ckpt = os.path.join(self.checkpoint_path, name+".pth")
            best_ckpt_abs_path = os.path.abspath(best_ckpt)
            if not os.path.exists(best_ckpt_abs_path):
                raise FileNotFoundError(f"Source file {best_ckpt_abs_path} does not exist.")
            os.symlink(best_ckpt_abs_path, best_symlink)

    def _save(self, name, model, model_options, common_options, train_options):
        obj = {}
        obj.update({"config": {"model_options": model_options, "common_options": common_options, "train_options": train_options}})
        obj.update(
            {
                "model_state_dict": model.state_dict(),
                "task": self.trainer.task,
                "optimizer_state_dict": self.trainer.optimizer.state_dict(), 
                "lr_scheduler_state_dict": self.trainer.lr_scheduler.state_dict(),
                "epoch": self.trainer.ep,
                "iteration":self.trainer.iter, 
                "stats": self.trainer.stats}
                )
        f_path = os.path.join(self.checkpoint_path, name+".pth")
        torch.save(obj, f=f_path)

            
        log.info(msg="checkpoint saved as {}".format(name))
