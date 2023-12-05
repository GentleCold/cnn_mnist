import time
import argparse

from train import Model

if __name__ == "__main__":
    # 解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", type=int,
                        default=64, help="batch size, defaul=64")
    parser.add_argument("-e", type=int, default=10,
                        help="number of epochs, default=10")
    parser.add_argument("-l", type=float,
                        default=0.001, help="learning rate, default=0.001")
    parser.add_argument("-d", type=float,
                        default=0.5, help="dropout rate, default=0.5")
    parser.add_argument("-m", type=str,
                        default="lenet", help="model type, default=lenet")
    parser.add_argument("-o", type=str,
                        default="adamw", help="optimizer type, default=adamw")

    args = parser.parse_args()
    start_time = time.time()
    model = Model(args.b, args.e, args.l, args.d, args.m, args.o)
    model.train()
    end_time = time.time()

    print("training time: ", end_time - start_time)
    model.draw_plt()
