from train import Trainer
import os
import sys

def main():
    index = 0
    args = sys.argv
    if len(args) > 1:
        if args[1].replace(',', '').replace('.', '').replace('-', '').isdigit():
            index = int(args[1])
        else:
            print("----------------------------------")
            print("")
            print("Argument is not digit")
            print("Set index to 0")
            print("")
            print("----------------------------------")
    else:
        print("----------------------------------")
        print("")
        print("Arguments are too short")
        print("Set index to 0")
        print("")
        print("----------------------------------")
    setting_csv_path = "./setting.csv"
    trainer = Trainer(setting_csv_path=setting_csv_path, index=index)
    if not os.path.isfile(os.path.join(trainer.log_path, trainer.model_name, trainer.model_name)):
        print("Trained weight file does not exist")
        trainer.train()

if __name__ == '__main__':
    main()
