from collections import defaultdict
from functools import partial
from pathlib import Path
import shutil
import sys
import time
import math
from typing import Any, Dict, Optional, Tuple

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

from agent import Agent
from collector import Collector
from envs import SingleProcessEnv, MultiProcessEnv
from episode import Episode
from make_reconstructions import make_reconstructions_from_batch
#from models.actor_critic import ActorCritic
from models.world_model import WorldModel
from utils import configure_optimizer, EpisodeDirManager, set_seed


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
        self.start_epoch = 1
        self.device = torch.device(cfg.common.device)
        self.batch_size=cfg.common.batch_size
        self.obs_time = cfg.common.obs_time 
        self.pred_time = cfg.common.pred_time 
        self.time_interval = cfg.common.time_interval
        
        ## where the check points should be saved no need to change this'

        self.ckpt_dir = Path('checkpoints')
        self.media_dir = Path('media')
        self.episode_dir = self.media_dir / 'episodes'
        self.reconstructions_dir = self.media_dir / 'reconstructions'

        if not cfg.common.resume:
            config_dir = Path('config')
            config_path = config_dir / 'trainer.yaml'
            config_dir.mkdir(exist_ok=False, parents=False)
            shutil.copy('.hydra/config.yaml', config_path)
            wandb.save(str(config_path))
            shutil.copytree(src=(Path(hydra.utils.get_original_cwd()) / "src"), dst="./src")
            shutil.copytree(src=(Path(hydra.utils.get_original_cwd()) / "scripts"), dst="./scripts")
            self.ckpt_dir.mkdir(exist_ok=False, parents=False)
            self.media_dir.mkdir(exist_ok=False, parents=False)
            self.episode_dir.mkdir(exist_ok=False, parents=False)
            self.reconstructions_dir.mkdir(exist_ok=False, parents=False)
        ################################################################

        ## Episode manager needed to be used later:  how many episodes to be saved and where 
        episode_manager_train = EpisodeDirManager(self.episode_dir / 'train', max_num_episodes=cfg.collection.train.num_episodes_to_save)
        episode_manager_test = EpisodeDirManager(self.episode_dir / 'test', max_num_episodes=cfg.collection.test.num_episodes_to_save)
        #self.episode_manager_imagination = EpisodeDirManager(self.episode_dir / 'imagination', max_num_episodes=cfg.evaluation.actor_critic.num_episodes_to_save)
        #################################################################


        def create_env(cfg_env, num_envs):
            env_fn = partial(instantiate, config=cfg_env)
            return MultiProcessEnv(env_fn, num_envs, should_wait_num_envs_ratio=1.0) if num_envs > 1 else SingleProcessEnv(env_fn)

        if self.cfg.training.should:
            train_env = create_env(cfg.env.train, cfg.collection.train.num_envs)
            self.train_dataset = instantiate(cfg.datasets.train)
            self.train_collector = Collector(train_env, self.train_dataset, episode_manager_train, obs_time=self.obs_time, pred_time=self.pred_time, time_interval=self.time_interval)

        if self.cfg.evaluation.should:
            test_env = create_env(cfg.env.test, cfg.collection.test.num_envs)
            self.test_dataset = instantiate(cfg.datasets.test)
            self.test_collector = Collector(test_env, self.test_dataset, episode_manager_test, obs_time=self.obs_time, pred_time=self.pred_time, time_interval=self.time_interval)

        assert self.cfg.training.should or self.cfg.evaluation.should
        env = train_env if self.cfg.training.should else test_env

        tokenizer = instantiate(cfg.tokenizer)
        world_model = WorldModel(obs_vocab_size=tokenizer.vocab_size, config=instantiate(cfg.world_model))
        # actor_critic = ActorCritic(**cfg.actor_critic, act_vocab_size=env.num_actions)
        self.agent = Agent(tokenizer, world_model).to(self.device)
        print(f'{sum(p.numel() for p in self.agent.tokenizer.parameters())} parameters in agent.tokenizer')
        print(f'{sum(p.numel() for p in self.agent.world_model.parameters())} parameters in agent.world_model')
        #print(f'{sum(p.numel() for p in self.agent.actor_critic.parameters())} parameters in agent.actor_critic')

        self.optimizer_tokenizer = torch.optim.Adam(self.agent.tokenizer.parameters(), lr=cfg.training.learning_rate)
        self.optimizer_world_model = configure_optimizer(self.agent.world_model, cfg.training.learning_rate, cfg.training.world_model.weight_decay)
        #self.optimizer_actor_critic = torch.optim.Adam(self.agent.actor_critic.parameters(), lr=cfg.training.learning_rate)

        if cfg.initialization.path_to_checkpoint is not None:
            self.agent.load(**cfg.initialization, device=self.device)

        if cfg.common.resume:
            self.load_checkpoint()

    def run(self) -> None:
        i=0

        for epoch in range(self.start_epoch, 1 + self.cfg.common.epochs):

            print(f"\nEpoch {epoch} / {self.cfg.common.epochs}\n")
            start_time = time.time()
            to_log = []

            if self.cfg.training.should:
                
                if epoch <= self.cfg.collection.train.stop_after_epochs:
                    batch=0
                    training_data, length_train =self.train_collector.collect_training_data()
                    #########################################################
                    nb_train_batches_per_epoch=math.ceil(length_train/self.batch_size)

                    to_log += self.train_agent(epoch, nb_train_batches_per_epoch, training_data)

            if self.cfg.evaluation.should and (epoch % self.cfg.evaluation.every == 0):
                self.test_dataset.clear()
                testing_data, length_test= self.test_collector.collect_testing_data()
                ####################################################
                nb_test_batches_per_epoch= math.ceil(length_test/self.batch_size)
                to_log += self.eval_agent(epoch, nb_test_batches_per_epoch, testing_data)

            if self.cfg.training.should:
                self.save_checkpoint(epoch, save_agent_only=not self.cfg.common.do_checkpoint)

            to_log.append({'duration': (time.time() - start_time) / 3600})
            for metrics in to_log:
                wandb.log({'epoch': epoch, **metrics})

        self.finish()

    def train_agent(self, epoch: int, nb_train_batches_per_epoch, training_data) -> None:
        self.agent.train()
        self.agent.zero_grad()
        

        metrics_tokenizer, metrics_world_model= {}, {}

        cfg_tokenizer = self.cfg.training.tokenizer
        cfg_world_model = self.cfg.training.world_model
        w = self.cfg.training.sampling_weights
        


        if cfg_tokenizer.start_after_epochs <= epoch <= cfg_tokenizer.stop_after_epochs:
            data_index=0
            loss_total_epoch = 0.0
            intermediate_losses = defaultdict(float)
            for _ in tqdm(range(nb_train_batches_per_epoch + 1), desc=f"Training {str(self.agent.tokenizer)}", file=sys.stdout):
                _, index =self.train_collector.get_next_batch(epoch, self.batch_size, data_index, training_data)
                data_index =index
                metrics_tokenizer, loss, intermediate_los = self.train_component(self.agent.tokenizer, self.optimizer_tokenizer,loss_total_epoch, intermediate_losses, sequence_length=self.cfg.common.sequence_length, sample_from_start=True, sampling_weights=w, **cfg_tokenizer) 
                loss_total_epoch = loss 
                intermediate_losses = intermediate_los
        self.agent.tokenizer.eval()

        if cfg_world_model.start_after_epochs <= epoch <= cfg_world_model.stop_after_epochs:
            data_index=0
            loss_total_epoch = 0.0
            intermediate_losses = defaultdict(float)
            for _ in tqdm(range(nb_train_batches_per_epoch + 1), desc=f"Training {str(self.agent.world_model)}", file=sys.stdout): 
                _, index =self.train_collector.get_next_batch(epoch, self.batch_size, data_index, training_data)
                data_index =index
                metrics_world_model, loss, intermediate_los = self.train_component(self.agent.world_model, self.optimizer_world_model, loss_total_epoch, sequence_length=self.cfg.common.sequence_length, sample_from_start=True, sampling_weights=w, tokenizer=self.agent.tokenizer, **cfg_world_model)
                loss_total_epoch = loss 
                intermediate_losses = intermediate_los
            #loss_total_epoch = loss/ (nb_train_batches_per_epoch +1)
            #_, index =self.train_collector.get_next_batch(epoch, self.batch_size, data_index, training_data)
            #metrics_world_model, loss = self.train_component(self.agent.world_model, self.optimizer_world_model, loss_total_epoch, sequence_length=self.cfg.common.sequence_length, sample_from_start=True, sampling_weights=w, tokenizer=self.agent.tokenizer, **cfg_world_model)
        self.agent.world_model.eval()

        # if epoch > cfg_actor_critic.start_after_epochs:
        #     metrics_actor_critic = self.train_component(self.agent.actor_critic, self.optimizer_actor_critic, sequence_length=1 + self.cfg.training.actor_critic.burn_in, sample_from_start=False, sampling_weights=w, tokenizer=self.agent.tokenizer, world_model=self.agent.world_model, **cfg_actor_critic)
        # self.agent.actor_critic.eval()

        return [{'epoch': epoch, **metrics_tokenizer, **metrics_world_model}]

    def train_component(self, component: nn.Module, optimizer: torch.optim.Optimizer, loss_total_epoch, intermediate_losses,  batch_num_samples: int, grad_acc_steps: int, sequence_length: int, sampling_weights: Optional[Tuple[float]], sample_from_start: bool, **kwargs_loss: Any) -> Dict[str, float]:
        
        #intermediate_losses = defaultdict(float)

        # for batch in tqdm(range(nb_batches_per_epoch+1), desc=f"Training {str(component)}", file=sys.stdout):
        optimizer.zero_grad()
            #batch, index_updated = self.train_collector.get_next_batch(epoch, batch_size, index, training_data)
        batch = self.train_dataset.batch_buffer(batch_num_samples, sequence_length, sampling_weights, sample_from_start)
        batch = self._to_device(batch)
        mini_batch= math.ceil(self.batch_size /(batch_num_samples*grad_acc_steps))
        counter=0
        for _ in range(mini_batch):
            for _ in range(grad_acc_steps): 
                batch_training= batch['observations'][:,:,(counter*batch_num_samples):(counter+1)*(batch_num_samples),:,:]
                losses = component.compute_loss(batch_training, **kwargs_loss) / grad_acc_steps
                loss_total_step = losses.loss_total
                loss_total_step.backward()
                loss_total_epoch += loss_total_step.item() 

                for loss_name, loss_value in losses.intermediate_losses.items():
                    intermediate_losses[f"{str(component)}/train/{loss_name}"] += loss_value
                
                counter= counter + 1


                # batch = self._to_device(batch)
                # losses = component.compute_loss(batch, **kwargs_loss)/grad_acc_steps
                # loss_total_step = losses.loss_total
                # (loss_total_step/batch_num_samples).backward()
                # loss_total_epoch += (loss_total_step).item()/batch_num_samples
                # index= index_updated
                

                # for loss_name, loss_value in losses.intermediate_losses.items():
                    # intermediate_losses[f"{str(component)}/train/{loss_name}"] += (loss_value/batch_num_samples) 

                # if max_grad_norm is not None:
                    # torch.nn.utils.clip_grad_norm_(component.parameters(), max_grad_norm)

            optimizer.step()

         
        metrics = {f'{str(component)}/train/total_loss': loss_total_epoch, **intermediate_losses}


        print("loss_total_epoch", loss_total_epoch)

        #batch = self._out_device(batch)
        return metrics, loss_total_epoch, intermediate_losses

    @torch.no_grad()
    def eval_agent(self, epoch: int, nb_test_batches_per_epoch, testing_data) -> None:
        self.agent.eval()

        metrics_tokenizer, metrics_world_model = {}, {}

        cfg_tokenizer = self.cfg.evaluation.tokenizer
        cfg_world_model = self.cfg.evaluation.world_model
        

        if epoch > cfg_tokenizer.start_after_epochs:
            test_data_index=0
            loss_total_test_epoch = 0.0
            for _ in tqdm(range(nb_test_batches_per_epoch + 1), desc=f"Evaluating {str(self.agent.tokenizer)}", file=sys.stdout):
                _, index =self.test_collector.get_next_batch(epoch, self.batch_size, test_data_index, testing_data)
                test_data_index =index
                metrics_tokenizer, loss_test = self.eval_component(self.agent.tokenizer, cfg_tokenizer.batch_num_samples, loss_total_test_epoch, sequence_length=self.cfg.common.sequence_length)
                loss_total_test_epoch = loss_test 
            # loss_total_test_epoch = loss_test/ (nb_test_batches_per_epoch + 1)
            # _, index =self.test_collector.get_next_batch(epoch, self.batch_size, test_data_index, testing_data)
            # metrics_tokenizer, loss_test = self.eval_component(self.agent.tokenizer, cfg_tokenizer.batch_num_samples, loss_total_test_epoch, sequence_length=self.cfg.common.sequence_length)
            

        if epoch > cfg_world_model.start_after_epochs:
            test_data_index=0
            loss_total_test_epoch = 0.0
            for _ in tqdm(range(nb_test_batches_per_epoch + 1), desc=f"Evaluating {str(self.agent.world_model)}", file=sys.stdout):
                _, index =self.test_collector.get_next_batch(epoch, self.batch_size, test_data_index, testing_data)
                test_data_index =index
                metrics_world_model, loss_test= self.eval_component(self.agent.world_model, cfg_world_model.batch_num_samples, loss_total_test_epoch, sequence_length=self.cfg.common.sequence_length, tokenizer=self.agent.tokenizer)
                loss_total_test_epoch = loss_test 
            # loss_total_test_epoch = loss_test/ (nb_test_batches_per_epoch +1)
            # _, index =self.test_collector.get_next_batch(epoch, self.batch_size, test_data_index, testing_data)
            # metrics_world_model, loss_test= self.eval_component(self.agent.world_model, cfg_world_model.batch_num_samples, loss_total_test_epoch, sequence_length=self.cfg.common.sequence_length, tokenizer=self.agent.tokenizer)


        if cfg_tokenizer.save_reconstructions:
            batch = self._to_device(self.test_dataset.batch_buffer(batch_num_samples=1, sequence_length=self.cfg.common.sequence_length))
            make_reconstructions_from_batch(batch, save_dir=self.reconstructions_dir, epoch=epoch, tokenizer=self.agent.tokenizer)

        return [metrics_tokenizer, metrics_world_model]

    @torch.no_grad()
    def eval_component(self, component: nn.Module, batch_num_samples: int, loss_total_test_epoch, sequence_length: int, **kwargs_loss: Any) -> Dict[str, float]:
        intermediate_losses = defaultdict(float)

        steps = 0
        #pbar = tqdm(desc=f"Evaluating {str(component)}", file=sys.stdout)
        for batch in self.test_dataset.traverse(batch_num_samples, sequence_length):
            batch = self._to_device(batch)
            batch_test = batch['observations'][:,:,0:(batch_num_samples),:,:]

            losses = component.compute_loss(batch_test, **kwargs_loss)
            loss_total_test_epoch += (losses.loss_total.item())

            for loss_name, loss_value in losses.intermediate_losses.items():
                intermediate_losses[f"{str(component)}/eval/{loss_name}"] += loss_value

            steps += 1
            #pbar.update(1)

        intermediate_losses = {k: v / steps for k, v in intermediate_losses.items()}
        metrics = {f'{str(component)}/eval/total_loss': loss_total_test_epoch, **intermediate_losses}
        return metrics, loss_total_test_epoch

    # @torch.no_grad()
    # def inspect_imagination(self, epoch: int) -> None:
    #     mode_str = 'imagination'
    #     batch = self.test_dataset.sample_batch(batch_num_samples=self.episode_manager_imagination.max_num_episodes, sequence_length=1 + self.cfg.training.actor_critic.burn_in, sample_from_start=False)
    #     outputs = self.agent.actor_critic.imagine(self._to_device(batch), self.agent.tokenizer, self.agent.world_model, horizon=self.cfg.evaluation.actor_critic.horizon, show_pbar=True)

    #     to_log = []
    #     for i, (o, a, r, d) in enumerate(zip(outputs.observations.cpu(), outputs.actions.cpu(), outputs.rewards.cpu(), outputs.ends.long().cpu())):  # Make everything (N, T, ...) instead of (T, N, ...)
    #         episode = Episode(o, a, r, d, torch.ones_like(d))
    #         episode_id = (epoch - 1 - self.cfg.training.actor_critic.start_after_epochs) * outputs.observations.size(0) + i
    #         self.episode_manager_imagination.save(episode, episode_id, epoch)

    #         metrics_episode = {k: v for k, v in episode.compute_metrics().__dict__.items()}
    #         metrics_episode['episode_num'] = episode_id
    #         metrics_episode['action_histogram'] = wandb.Histogram(episode.actions.numpy(), num_bins=self.agent.world_model.act_vocab_size)
    #         to_log.append({f'{mode_str}/{k}': v for k, v in metrics_episode.items()})

    #     return to_log

    def _save_checkpoint(self, epoch: int, save_agent_only: bool) -> None:
        torch.save(self.agent.state_dict(), self.ckpt_dir / 'last.pt')
        if not save_agent_only:
            torch.save(epoch, self.ckpt_dir / 'epoch.pt')
            torch.save({
                "optimizer_tokenizer": self.optimizer_tokenizer.state_dict(),
                "optimizer_world_model": self.optimizer_world_model.state_dict(),
            }, self.ckpt_dir / 'optimizer.pt')
            # ckpt_dataset_dir = self.ckpt_dir / 'dataset'
            # ckpt_dataset_dir.mkdir(exist_ok=True, parents=False)
            # self.train_dataset.update_disk_checkpoint(ckpt_dataset_dir)
            if self.cfg.evaluation.should:
                torch.save(self.test_dataset.num_seen_episodes, self.ckpt_dir / 'num_seen_episodes_test_dataset.pt')

    def save_checkpoint(self, epoch: int, save_agent_only: bool) -> None:
        tmp_checkpoint_dir = Path('checkpoints_tmp')
        shutil.copytree(src=self.ckpt_dir, dst=tmp_checkpoint_dir, ignore=shutil.ignore_patterns('dataset'))
        self._save_checkpoint(epoch, save_agent_only)
        shutil.rmtree(tmp_checkpoint_dir)

    def load_checkpoint(self) -> None:
        assert self.ckpt_dir.is_dir()
        self.start_epoch = torch.load(self.ckpt_dir / 'epoch.pt') + 1
        self.agent.load(self.ckpt_dir / 'last.pt', device=self.device)
        ckpt_opt = torch.load(self.ckpt_dir / 'optimizer.pt', map_location=self.device)
        self.optimizer_tokenizer.load_state_dict(ckpt_opt['optimizer_tokenizer'])
        self.optimizer_world_model.load_state_dict(ckpt_opt['optimizer_world_model'])
        self.train_dataset.load_disk_checkpoint(self.ckpt_dir / 'dataset')
        if self.cfg.evaluation.should:
            self.test_dataset.num_seen_episodes = torch.load(self.ckpt_dir / 'num_seen_episodes_test_dataset.pt')
        print(f'Successfully loaded model, optimizer and {len(self.train_dataset)} episodes from {self.ckpt_dir.absolute()}.')

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: batch[k].to(self.device) for k in batch}

    def _out_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: batch[k].detach() for k in batch}
    
    def finish(self) -> None:
        wandb.finish()
