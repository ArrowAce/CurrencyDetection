import os
from os import path
import shutil

Source_Path = r'D:\Users\ajayshankar\Desktop\AJAY\Currency Detection\New folder\New folder\Indian Currencies zip\Indian Currencies\all'
Destination = r'D:\Users\ajayshankar\Desktop\AJAY\Currency Detection\New folder\New folder\Indian Currencies zip\Indian Currencies_dest'
#dst_folder = os.mkdir(Destination)


def main():
    for count, filename in enumerate(os.listdir(Source_Path)):
        n= count+1
        if n<10:

            dst =  "INDIA0100_00" + str(n) + ".jpg"

        elif n>=10 and n<100:
            dst =  "INDIA0100_0" + str(n) + ".jpg"

        else:
            dst= "INDIA0100_" + str(n)+ ".jpg"


        # rename all the files
        os.rename(os.path.join(Source_Path, filename),  os.path.join(Destination, dst))


# Driver Code
if __name__ == '__main__':
    main()