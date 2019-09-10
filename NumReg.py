import argparse, textwrap
import lib.generate as g
import lib.recognize as r

def main():
    print('''
    __  __                      ____                       
    /\ \/\ \                    /\  _`\                     
    \ \ `\\ \  __  __    ___ ___\ \ \L\ \     __     __     
    \ \ , ` \/\ \/\ \ /' __` __`\ \ ,  /   /'__`\ /'_ `\   
    \ \ \`\ \ \ \_\ \/\ \/\ \/\ \ \ \\ \ /\  __//\ \L\ \  
    \ \_\ \_\ \____/\ \_\ \_\ \_\ \_\ \_\ \____\ \____ \ 
        \/_/\/_/\/___/  \/_/\/_/\/_/\/_/\/ /\/____/\/___L  |
                                                    /\___//
                                                    \_/__/ 
                                                    
                                    --version:1.0 Evi1ran
                                    ''')
    parser = argparse.ArgumentParser(description="Handwritten Numbers Recognition", usage='use "python NumReg.py -h" for more information', formatter_class=argparse.RawTextHelpFormatter)
    group = parser.add_mutually_exclusive_group(required = True)
    group.add_argument("-g", type = int, default = '10000', help='generate handwritten numbers database (default generate 10000 pics)')
    group.add_argument("-r", help='recognize handwritten number')
    parser.add_argument("-s",default='msyh.ttc', help= 'handwritten numbers database source font (default font is lib/msyh.ttc)')
    parser.add_argument("-f", "--flag", type = int, choices = [1, 2, 3, 4], default='1', help= textwrap.dedent('''\
            1. LR，Logistic Regression 
            2. Linear SVC，Support Vector Classification
            3. MLPC，Multi-Layer Perceptron Classification 
            4. SGDC，Stochastic Gradient Descent Classification'''))
    args = parser.parse_args()
    if args.r is None:
        try:
            g.generateNums(args.g, args.s)
        except:
            print("[Error occurred]")
    else:
        r.recognizeNumber(args.r, args.flag)

if __name__ == "__main__":
    main()

