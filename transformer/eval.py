import torch

from config import get_config
from dataset import dataset
from train import MainModel

config = get_config()

def predict(model, text):
    text = dataset.create_test_data(text).cuda()
    text = model(text, use_mask=True)
    text = torch.argmax(text, dim=-1)[0].tolist()
    text = dataset.vocab.lookup_tokens(text)
    return text

def generate(model, text):
    next_token = predict(model, text)[-1]
    text += f' {next_token}'
    return text

def prepare(model):
    dataset._get_vocab()
    model.eval()

def is_valid_input(text: str) -> bool:
    n_input_tokens = len(dataset.tokenize(text)[0])
    return n_input_tokens < config.max_seq_len and n_input_tokens > 0

def get_input() -> str:
    prompt_text = f'Enter the text to do next word prediction on (max {config.max_seq_len - 1} words): '
    text = input(prompt_text)

    while not is_valid_input(text):
        print(f'Can only accept between 1 and {config.max_seq_len - 1} words')
        text = input(prompt_text)

    return text

def should_exit() -> bool:
    prompt_text = f'Do you want to try another prediction (Y/n) '

    valid_parameter_entered = False
    exit_value = None

    while not valid_parameter_entered:
        text = input(prompt_text)
        if text is None or text.strip() == '':
            exit_value = False
            valid_parameter_entered = True
        elif text == 'y' or text == 'Y':
            exit_value = False
            valid_parameter_entered = True
        elif text == 'n' or text == 'N':
            exit_value = True
            valid_parameter_entered = True
        else:
            print('Invalid answer please enter "y" or "n"')

    return exit_value

if __name__ == '__main__':
    PATH = '' # insert path here
    if PATH == '':
        print('Please specify the path to model in this script')
        exit()

    print('Loading model and vocab...')
    model = MainModel.load_from_checkpoint(PATH, n_tokens=226_798).cuda()
    model = model.model
    prepare(model)

    done = False
    while not done:
        text = get_input()

        for i in range(config.max_seq_len - len(text)):
            text = generate(model, text)

        print('Predicted text:')
        print(text)
        done = should_exit()
