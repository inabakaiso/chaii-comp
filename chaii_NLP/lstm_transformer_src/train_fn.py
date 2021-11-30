from utils import *
from config import *
from train import *


class Trainer:
    def __init__(
            self, model, tokenizer,
            optimizer, scheduler
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(
            self, args,
            train_dataloader,
            epoch, result_dict
    ):
        count = 0
        losses = AverageMeter()

        self.model.zero_grad()
        self.model.train()

        fix_all_seeds(args.seed)
        for batch_idx, batch_data in enumerate(train_dataloader):
            input_ids, attention_mask, targets_start, targets_end = \
                batch_data['input_ids'], batch_data['attention_mask'], \
                batch_data['start_position'], batch_data['end_position']

            input_ids, attention_mask, targets_start, targets_end = \
                input_ids.cuda(), attention_mask.cuda(), targets_start.cuda(), targets_end.cuda()

            outputs_start, outputs_end = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss = loss_fn((outputs_start, outputs_end), (targets_start, targets_end))
            loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            count += input_ids.size(0)
            losses.update(loss.item(), input_ids.size(0))

            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)

            if batch_idx % args.gradient_accumulation_steps == 0 or batch_idx == len(train_dataloader) - 1:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            if (batch_idx % args.logging_steps == 0) or (batch_idx + 1) == len(train_dataloader):
                _s = str(len(str(len(train_dataloader.sampler))))
                ret = [
                    ('Epoch: {:0>2} [{: >' + _s + '}/{} ({: >3.0f}%)]').format(epoch, count,
                                                                               len(train_dataloader.sampler),
                                                                               100 * count / len(
                                                                                   train_dataloader.sampler)),
                    'Train Loss: {: >4.5f}'.format(losses.avg),
                ]
                print(', '.join(ret))

        result_dict['train_loss'].append(losses.avg)
        return result_dict


class Evaluator:
    def __init__(self, model):
        self.model = model

    def save(self, result, output_dir):
        with open(f'{output_dir}/result_dict.json', 'w') as f:
            f.write(json.dumps(result, sort_keys=True, indent=4, ensure_ascii=False))

    def evaluate(self, valid_dataloader, epoch, result_dict):
        losses = AverageMeter()
        all_outputs_start, all_outputs_end = [], []
        for batch_idx, batch_data in enumerate(valid_dataloader):
            self.model = self.model.eval()
            input_ids, attention_mask, targets_start, targets_end = \
                batch_data['input_ids'], batch_data['attention_mask'], \
                batch_data['start_position'], batch_data['end_position']

            input_ids, attention_mask, targets_start, targets_end = \
                input_ids.cuda(), attention_mask.cuda(), targets_start.cuda(), targets_end.cuda()

            with torch.no_grad():
                outputs_start, outputs_end = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                all_outputs_start.append(outputs_start.cpu().numpy().tolist())
                all_outputs_end.append(outputs_end.cpu().numpy().tolist())

                loss = loss_fn((outputs_start, outputs_end), (targets_start, targets_end))
                losses.update(loss.item(), input_ids.size(0))

        all_outputs_start = np.vstack(all_outputs_start)
        all_outputs_end = np.vstack(all_outputs_end)

        print('----Validation Results Summary----')
        print('Epoch: [{}] Valid Loss: {: >4.5f}'.format(epoch, losses.avg))
        result_dict['val_loss'].append(losses.avg)
        return result_dict, all_outputs_start, all_outputs_end