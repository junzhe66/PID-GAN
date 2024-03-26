from collections import defaultdict
from pathlib import Path
import shutil
import sys
import time
import math
from typing import Any, Dict

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

from agent import Agent
from collector import Collector
from make_reconstructions import make_reconstructions_from_batch, generate_reconstructions_with_tokenizer,compute_metrics
from make_prediction import compute_metrics_pre
from models.world_model import WorldModel
from utils import configure_optimizer, set_seed
from models.tokenizer import NLayerDiscriminator
from phy import RQ

class Trainer:
    def __init__(self, cfg: DictConfig) -> None:
        wandb.init(
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True,
            resume=True,
            **cfg.wandb
        )

        if cfg.common.seed is not None:
            set_seed(cfg.common.seed)

        self.cfg = cfg
        self.start_epoch = 23
        self.device = torch.device(cfg.common.device)
        self.batch_size=cfg.common.batch_size
        self.obs_time = cfg.common.obs_time 
        self.pred_time = cfg.common.pred_time
        self.optimizer_filename = cfg.checkpoint_OPT.name_to_checkpoint
        self.test_batch=cfg.evaluation.batch

        self.ckpt_dir = Path('/space/junzheyin/check_final_new1')
        self.media_dir = Path('media')
        self.reconstructions_dir = self.media_dir / 'reconstructions'

        if not cfg.common.resume:
            config_dir = Path('config')
            config_path = config_dir / 'trainer.yaml'
            config_dir.mkdir(exist_ok=False, parents=False)
            shutil.copy('.hydra/config.yaml', config_path)
            wandb.save(str(config_path))
            shutil.copytree(src=(Path(hydra.utils.get_original_cwd()) / "src"), dst="./src")
            self.ckpt_dir.mkdir(exist_ok=True, parents=False)
            self.media_dir.mkdir(exist_ok=False, parents=False)
            self.reconstructions_dir.mkdir(exist_ok=False, parents=False)
        ##################################################################

        if self.cfg.training.should:
            self.train_collector = Collector()

        if self.cfg.evaluation.should:
            self.test_collector = Collector()

        assert self.cfg.training.should or self.cfg.evaluation.should


        tokenizer = instantiate(cfg.tokenizer)
        world_model = WorldModel(obs_vocab_size=tokenizer.vocab_size, config=instantiate(cfg.world_model))
        discriminator = NLayerDiscriminator(input_nc=1, ndf=128, n_layers=3, use_actnorm=False, use_dropout=True, dropout_prob=0.1, noise_std=0)
        discriminator_AENN = world_model.head_discriminator.to(self.device)
        self.agent = Agent(tokenizer, world_model, discriminator, discriminator_AENN).to(self.device)

        print(f'{sum(p.numel() for p in self.agent.tokenizer.parameters())} parameters in agent.tokenizer')
        print(f'{sum(p.numel() for p in self.agent.world_model.parameters())} parameters in agent.world_model')
        print(f'{sum(p.numel() for p in self.agent.discriminator.parameters())} parameters in agent.discriminator')
        print(f'{sum(p.numel() for p in self.agent.discriminator_AENN.parameters())} parameters in agent.discriminator_AENN')
    

        self.optimizer_tokenizer = torch.optim.Adam(self.agent.tokenizer.parameters(), lr=cfg.training.learning_rate)
        self.optimizer_discriminator = torch.optim.Adam(self.agent.discriminator.parameters(), lr=0.00001)
        self.optimizer_world_model = configure_optimizer(self.agent.world_model, cfg.training.learning_rate, cfg.training.world_model.weight_decay)
        self.optimizer_discriminator_AENN = configure_optimizer(self.agent.discriminator_AENN, cfg.training.learning_rate, cfg.training.world_model.weight_decay)

        if cfg.initialization.path_to_checkpoint is not None:
            self.agent.load(**cfg.initialization, device=self.device)
        if cfg.checkpoint_OPT.load_opti:
            self.load_checkpoint()


    def run(self) -> None:

        training_data_dataloader = self.train_collector.collect_training_data(self.batch_size)
        testing_data_dataloader, length_test = self.test_collector.collect_ext_data(batch_size=1)



        for epoch in range(self.start_epoch, 1 + self.cfg.common.epochs):

            print(f"\nEpoch {epoch} / {self.cfg.common.epochs}\n")
            start_time = time.time()
            to_log = []

            if self.cfg.training.should:
                if epoch <= self.cfg.collection.train.stop_after_epochs:
                    to_log += self.train_agent(epoch, training_data_dataloader)

            if self.cfg.evaluation.should and (epoch % self.cfg.evaluation.every == 0):
                to_log += self.eval_agent(epoch, testing_data_dataloader, length_test)

            if self.cfg.training.should:
                self.save_checkpoint(epoch, save_agent_only=not self.cfg.common.do_checkpoint)

            to_log.append({'duration': (time.time() - start_time) / 3600})
            for metrics in to_log:
                wandb.log({'epoch': epoch, **metrics})

        self.finish()


    def train_agent(self, epoch: int, training_data_dataloader) -> None:
        self.agent.train()
        self.agent.zero_grad()
        

        metrics_tokenizer, metrics_world_model, metrics_discriminator= {}, {}, {}

        cfg_tokenizer = self.cfg.training.tokenizer
        cfg_world_model = self.cfg.training.world_model
        cfg_discriminator = self.cfg.training.discriminator

        if epoch >= cfg_tokenizer.start_after_epochs:
            loss_total_epoch = 0.0
            intermediate_losses = defaultdict(float)
            for batch in training_data_dataloader:
                batch= batch.unsqueeze(2)
                metrics_tokenizer, loss, intermediate_los = self.train_component(self.agent.tokenizer, self.optimizer_tokenizer,batch, loss_total_epoch, intermediate_losses, sequence_length=self.cfg.common.sequence_length, **cfg_tokenizer) 
                loss_total_epoch = loss 
                intermediate_losses = intermediate_los
            print("tokenizer_loss_total_epoch", loss_total_epoch)
        self.agent.tokenizer.eval()



        # if epoch >= cfg_world_model.start_after_epochs:
        #     loss_total_epoch = 0.0
        #     intermediate_losses = defaultdict(float)
        #     for (phy_data, radar_data) in training_data_dataloader:
        #         batch= radar_data.unsqueeze(2)
        #         metrics_world_model, loss, intermediate_los = self.train_component(self.agent.world_model, self.optimizer_world_model, batch, loss_total_epoch, intermediate_losses, tokenizer=self.agent.tokenizer, train_world_model=True, **cfg_world_model)
        #         loss_total_epoch = loss 
        #         intermediate_losses = intermediate_los
        #     print("worldmodel_loss_total_epoch", loss_total_epoch)
        # self.agent.world_model.eval()


        # if epoch >= cfg_discriminator.start_after_epochs:
        #     loss_total_epoch = 0.0
        #     intermediate_losses = defaultdict(float)
                    
        #     for (phy_data, radar_data) in training_data_dataloader:
        #         batch= radar_data.unsqueeze(2)
        #         metrics_discriminator, loss, intermediate_los = self.train_component(self.agent.world_model, self.optimizer_discriminator_AENN, batch, loss_total_epoch, intermediate_losses, tokenizer=self.agent.tokenizer, train_world_model=False, **cfg_discriminator)
        #         loss_total_epoch = loss 
        #         intermediate_losses = intermediate_los
        #         print("discriminator_total_loss", loss_total_epoch)
        # self.agent.world_model.eval()


        
        if epoch >= cfg_world_model.start_after_epochs:
            loss_total_epoch_G = 0.0
            loss_total_epoch_D = 0.0
            intermediate_losses_G = defaultdict(float)
            for (phy_data, radar_data) in training_data_dataloader:
                batch= radar_data.unsqueeze(2)
                Q=RQ(phy_data)
                Q=Q.unsqueeze(2)
                #steps_per_epoch = math.floor(length_train / batch.size(0) )
                #print(steps_per_epoch)
                metrics_world_model, loss_G, loss_D ,intermediate_los_G = self.train_component_GAN(self.agent.world_model, self.optimizer_world_model, self.agent.world_model, self.optimizer_discriminator_AENN, batch, Q, loss_total_epoch_G, intermediate_losses_G, loss_total_epoch_D, tokenizer=self.agent.tokenizer, **cfg_world_model) 
                loss_total_epoch_G = loss_G
                loss_total_epoch_D = loss_D

                intermediate_losses_G = intermediate_los_G

            print("worldmodel_loss_total_epoch", loss_total_epoch_G)
            print("discriminator_total_loss", loss_total_epoch_D)
        self.agent.world_model.eval()

        return [{'epoch': epoch, **metrics_tokenizer, **metrics_world_model, **metrics_discriminator}]
    

    def train_component_GAN(self, component_G, optimizer_G, component_D, optimizer_D, batch, phy_data, loss_total_epoch_G, intermediate_losses_G, loss_total_epoch_D, batch_num_samples, tokenizer,**kwargs_loss):
        mini_batch = math.floor(batch.size(0) / batch_num_samples)
        counter = 0

        for i in range(mini_batch):
            batch_training= batch[(counter*batch_num_samples):(counter+1)*(batch_num_samples),:,:,:,:]
            phy_data_training = phy_data[(counter*batch_num_samples):(counter+1)*(batch_num_samples),:,:,:,:]
            batch_training = self._to_device(batch_training)
            phy_data_training=self._to_device(phy_data_training)
            # Compute loss for G and D
            losses_G = component_G.compute_loss(batch_training, phy_data_training, tokenizer, train_world_model=True, **kwargs_loss) /64
            losses_D = component_D.compute_loss(batch_training, phy_data_training, tokenizer, train_world_model=False, **kwargs_loss) /64
            loss_total_step_G = losses_G.loss_total
            loss_total_step_D = losses_D.loss_total
            # Conditionally update D and G
            if 0 < (i + 1) % 128 < 65:  # Update D in first half of 128 iterations
                loss_total_step_D.backward()
                loss_total_epoch_D += loss_total_step_D.item()   # Log D loss
                print("Dis_loss", loss_total_epoch_D)
                if (i + 1) % 64 == 0:  # Update every 64 iterations
                    optimizer_D.step()
                    optimizer_D.zero_grad()
            else:  # Update G in second half of 128 iterations
                loss_total_step_G.backward()
                loss_total_epoch_G += loss_total_step_G.item()  # Log G loss
                for loss_name, loss_value in  losses_G.intermediate_losses.items():
                     intermediate_losses_G[f"{str(component_G)}/train/{loss_name}"] += loss_value 
                print("world_model_loss", loss_total_epoch_G)
                if (i + 1) % 64 == 0:  # Update every 64 iterations
                    optimizer_G.step()
                    optimizer_G.zero_grad()

            counter += 1
        # loss_total_epoch_G /= steps_per_epoch
        # loss_total_epoch_D /= steps_per_epoch

        # for loss_name in intermediate_losses_G:
        #     intermediate_losses_G[loss_name] /= steps_per_epoch

        metrics = {f'{str(component_G)}/train/total_loss': loss_total_epoch_G, **intermediate_losses_G}
        return metrics, loss_total_epoch_G, loss_total_epoch_D, intermediate_losses_G

    # def train_component(self, component: nn.Module, optimizer: torch.optim.Optimizer, batch,  loss_total_epoch, intermediate_losses,  batch_num_samples: int, grad_acc_steps: int, train_world_model: bool, **kwargs_loss: Any) -> Dict[str, float]:
    #     mini_batch= math.floor(batch.size(0)/(batch_num_samples*grad_acc_steps))
    #     counter=0

    #     for _ in range(mini_batch):
    #         optimizer.zero_grad()
    #         for _ in range(grad_acc_steps): 
    #             batch_training= batch[(counter*batch_num_samples):(counter+1)*(batch_num_samples),:,:,:,:] 
    #             batch_training = self._to_device(batch_training)
    #             losses = component.compute_loss(batch_training,train_world_model=train_world_model, **kwargs_loss) / grad_acc_steps
    #             loss_total_step = losses.loss_total
    #             loss_total_step.backward()
    #             loss_total_epoch += loss_total_step.item() 

    #             for loss_name, loss_value in losses.intermediate_losses.items():
    #                 intermediate_losses[f"{str(component)}/train/{loss_name}"] += loss_value
                
    #             counter= counter + 1
                
    #         print("loss_total_batch", loss_total_epoch)

    #         optimizer.step()

    #     if train_world_model==True:
    #        component1="world_model"
    #     else:
    #        component1="discriminator_AENN"


    #     metrics = {f'{str(component1)}/train/total_loss': loss_total_epoch, **intermediate_losses}
    #     return metrics, loss_total_epoch, intermediate_losses
    

    @torch.no_grad()
    def eval_agent(self, epoch: int, testing_data_dataloader, length_test) -> None:
        self.agent.eval()

        
        metrics_tokenizer, metrics_world_model= {}, {}

        cfg_tokenizer = self.cfg.training.tokenizer
        cfg_world_model = self.cfg.training.world_model

        
# 
        if epoch >= cfg_tokenizer.start_after_epochs:
            loss_total_test_epoch = 0.0
            intermediate_losses = defaultdict(float)
            self.accumulated_metrics = defaultdict(float)           
            for batch in testing_data_dataloader:
                batch= batch.unsqueeze(2)            
                metrics_tokenizer, loss_test, intermediate_los  = self.eval_component(self.agent.tokenizer, self.agent.discriminator, batch, loss_total_test_epoch, intermediate_losses)
                loss_total_test_epoch = loss_test 
                intermediate_losses = intermediate_los
                print("evaluation total loss", loss_total_test_epoch)
                
            for metrics_name, metrics_value in metrics_tokenizer.items():
                metrics_tokenizer[metrics_name] = metrics_value / length_test

        
        if epoch >= cfg_world_model.start_after_epochs:
            loss_total_test_epoch = 0.0
            intermediate_losses = defaultdict(float)
            self.accumulated_metrics = defaultdict(float)           
            for (phy_data, radar_data) in testing_data_dataloader:
                batch= radar_data.unsqueeze(2)
                Q=RQ(phy_data)
                Q=Q.unsqueeze(2)
                #print(batch.size())
                #batch = self._to_device(batch)
                #self.start_generation(generate_batch, epoch=epoch)
                metrics_world_model, loss_test, intermediate_los = self.eval_component(self.agent.world_model, self.agent.tokenizer,  batch, Q, loss_total_test_epoch, intermediate_losses, tokenizer=self.agent.tokenizer, train_world_model=True)
                loss_total_test_epoch = loss_test 
                intermediate_losses = intermediate_los
                print("evaluation total loss world model", loss_total_test_epoch)
            for metrics_name, metrics_value in metrics_world_model.items():
                metrics_world_model[metrics_name] = metrics_value / length_test


        # if epoch >= cfg_discriminator.start_after_epochs:
        #     loss_total_test_epoch = 0.0
        #     intermediate_losses = defaultdict(float)
                    
        #     for (phy_data, radar_data) in testing_data_dataloader:
        #         batch= radar_data.unsqueeze(2)
        #         #batch = self._to_device(batch)
        #         #self.start_generation(generate_batch, epoch=epoch)
        #         metrics_discriminator, loss_test, intermediate_los = self.eval_component(self.agent.world_model, self.agent.tokenizer, cfg_discriminator.batch_num_samples, batch, loss_total_test_epoch, intermediate_losses, sequence_length=self.cfg.common.sequence_length, tokenizer=self.agent.tokenizer, train_world_model=False)
        #         loss_total_test_epoch = loss_test 
        #         intermediate_losses = intermediate_los
        #         print("evaluation total loss discriminator", loss_total_test_epoch)


        return [metrics_tokenizer, metrics_world_model]
    
    

    @torch.no_grad()
    def eval_component(self, component: nn.Module, component1: nn.Module, batch, phy_data, loss_total_test_epoch, intermediate_losses, train_world_model: bool, **kwargs_loss: Any) -> Dict[str, float]:
        pysteps_metrics = {}
        
        batch_testing = self._to_device(batch)
        phy_data = self._to_device(phy_data)          
        losses = component.compute_loss(batch_testing, phy_data,**kwargs_loss, train_world_model=train_world_model)
        loss_total_test_epoch += (losses.loss_total.item())
        for loss_name, loss_value in losses.intermediate_losses.items():
            intermediate_losses[f"{str(component)}/eval/{loss_name}"] += loss_value
        
        if train_world_model == True:

            pysteps_metrics = compute_metrics_pre(batch_testing, component1, component)
        
            for metrics_name, metrics_value in pysteps_metrics.items():
                if math.isnan(metrics_value):
                    metrics_value = 0.0
                self.accumulated_metrics[metrics_name] += metrics_value
        

            intermediate_losses = {k: v  for k, v in intermediate_losses.items()}
            metrics = {f'{str(component)}/eval/total_loss': loss_total_test_epoch, **intermediate_losses, **self.accumulated_metrics}
        else: 
            intermediate_losses = {k: v  for k, v in intermediate_losses.items()}
            metrics = {"discriminator_AENN/eval/total_loss": loss_total_test_epoch, **intermediate_losses}

        # print("evaluation total loss", loss_total_test_epoch)

        return metrics, loss_total_test_epoch, intermediate_losses
    

    # @torch.no_grad()
    # def start_generation(self, batch, epoch) -> None:
        # predicted_observations= GenerationPhase.generate(self, batch, tokenizer= self.agent.tokenizer,world_model= self.agent.world_model,latent_dim=16, horizon=8, obs_time=8)
        # observations= batch[7:16,:,:,:]
        # GenerationPhase.show_prediction(observations,predictions, save_dir=self.generation_dir, epoch=epoch)





    def _save_checkpoint(self, epoch: int, save_agent_only: bool) -> None:
        if epoch % self.cfg.evaluation.every == 0:
           # torch.save(self.agent.state_dict(), self.ckpt_dir / f'model_checkpoint_epoch_{epoch:02d}.pt')
                   # Save only the world_model and discriminator_AENN parts of the agent
            state_to_save = {
                'world_model': self.agent.world_model.state_dict(),
                'discriminator_AENN': self.agent.discriminator_AENN.state_dict()
            }
            torch.save(state_to_save, self.ckpt_dir / f'model_checkpoint_epoch_{epoch:02d}.pt')
            if not save_agent_only:
                torch.save({
                    "optimizer_world_model": self.optimizer_world_model.state_dict(),
                    "optimizer_discriminator_AENN": self.optimizer_discriminator_AENN.state_dict(),
                }, self.ckpt_dir / f'optimizer_{epoch:02d}.pt')

    def save_checkpoint(self, epoch: int, save_agent_only: bool) -> None:
        self._save_checkpoint(epoch, save_agent_only)


    def load_checkpoint(self) -> None:
        assert self.ckpt_dir.is_dir()
        ckpt_opt = torch.load(self.ckpt_dir / self.optimizer_filename, map_location=self.device)
        self.optimizer_world_model.load_state_dict(ckpt_opt['optimizer_world_model'])
        self.optimizer_discriminator_AENN.load_state_dict(ckpt_opt['optimizer_discriminator_AENN'])
        print(f'Successfully loaded optimizer from {self.ckpt_dir.absolute()}.')

    def _to_device(self, batch: torch.Tensor):
        return batch.to(self.device)

    def _out_device(self, batch: torch.Tensor):
        return batch.detach()
    
    def finish(self) -> None:
        wandb.finish()