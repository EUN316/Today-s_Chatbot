import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
from models.tools import load_config
from system import QnA_system

def main():
    # Load config dict
    config_path = "./Base.yaml"
    cfg = load_config(config_path)
    # print('load_cfg')
    
    # Argparse
    parser = argparse.ArgumentParser(description='원하는 정보를 입력해주세요.')
    parser.add_argument('--product_id', required=True, help='상품 ID를 입력하세요.')
    parser.add_argument('--question', required=True, help='문의할 내용을 입력하세요.')
    
    args = parser.parse_args()
    # print(args.product_id)
    # print(args.question)
    
    # Run model
    # print('start running model')
    QnA_system(cfg, args.product_id, args.question)
    # print('end running model')

    
if __name__ == "__main__":
    main()