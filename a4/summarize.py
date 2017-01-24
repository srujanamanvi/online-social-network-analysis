"""
sumarize.py
"""
import cluster
import collect
import classify


def main():
    f1 = open('summary.txt','w')
    f = open('users.txt', 'r')
    f1.write(f.read())
    f.close()
    f = open('messages.txt','r')
    f1.write(f.read())
    f.close()
    f = open('community.txt','r')
    f1.write(f.read())
    f.close()
    f = open('class.txt','r')
    f1.write(f.read())
    f.close()
    f = open('example.txt','r')
    f1.write(f.read())
    f.close()
    f1.close()
    

if __name__ == '__main__':
    main()

