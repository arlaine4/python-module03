from ScrapBooker import ScrapBooker
import numpy as np


if __name__=="__main__":
    spb = ScrapBooker()
#################### CROP #########################################

    arr1 = np.arange(0,25).reshape(5,5)
    print(f"arr1 -->\n{arr1}")

    print('- - - - - - - - -')

    tmp = spb.crop(arr1, (3,1), (1,0))
    print(f"try to crop with (3,1) and (1, 0)")
    print(f"spb.crop -->\n{tmp}")
    #array([[5], [10], [15]])

    print('- - - - - - - - -')

    tmp = spb.crop(arr1, (1,1), (66, 42))
    print(f"try to crop with (1,1) and (66, 42)")
    print(f"spb.crop -->\n{tmp}")
    #None

    print('- - - - - - - - -')

    tmp = spb.crop(arr1, (0,0), (2, 2))
    print(f"try to crop with (0,0) and (2,2)")
    print(f"spb.crop -->\n{tmp}")
    #[]

    print('- - - - - - - - -')

    tmp = spb.crop(arr1, (1,1), (2, 2))
    print(f"try to crop with (1,1) and (2,2)")
    print(f"spb.crop -->\n{tmp}")
    #array([[12]])


##################################################################









#################### THIN #########################################

    print('-------------------------------------------------------------------')
    arr2 = np.array("A B C D E F G H I".split() * 6).reshape(-1,9)
    print(f"arr2 -->\n{arr2}")

    print('- - - - - - - - -')

    tmp = spb.thin(arr2, 3, 0)
    print(f"try to thin with 3, 0")
    print(f"thin ---> \n{tmp}")
    #array([[’A’, ’B’, ’D’, ’E’, ’G’, ’H’, ’J’, ’K’],
    #       [’A’, ’B’, ’D’, ’E’, ’G’, ’H’, ’J’, ’K’],
    #       [’A’, ’B’, ’D’, ’E’, ’G’, ’H’, ’J’, ’K’],
    #       [’A’, ’B’, ’D’, ’E’, ’G’, ’H’, ’J’, ’K’],
    #       [’A’, ’B’, ’D’, ’E’, ’G’, ’H’, ’J’, ’K’],
    #       [’A’, ’B’, ’D’, ’E’, ’G’, ’H’, ’J’, ’K’]], dtype=’<U1’)


    print('- - - - - - - - -')

    tmp = spb.thin(arr2, 2, 0)
    print(f"try to thin with 2, 0")
    print(f"thin ---> \n{tmp}")
    #array([[’A’, ’C’, ’E’,  ’G’, ’I’],
    #       [’A’, ’C’, ’E’,  ’G’, ’I’],
    #       [’A’, ’C’, ’E’,  ’G’, ’I’],
    #       [’A’, ’C’, ’E’,  ’G’, ’I’],
    #       [’A’, ’C’, ’E’,  ’G’, ’I’],

    print('- - - - - - - - -')

    tmp = spb.thin(arr2, 1, 0)
    print(f"try to thin with 1, 0")
    print(f"thin ---> \n{tmp}")
    #array([[]])

    print('-------------------------------------------------------------------')


##################################################################


#################### JUXTAPOSE #########################################

    arr3 = np.array([[1, 2, 3],[3, 2, 1],[1, 1, 1]])
    print(f"arr3 -->\n{arr3}")

    print(f"try to juxtapose with 3, 1")
    tmp = spb.juxtapose(arr3, 3, 1)
    print(f"juxtapose --->\n{tmp}")

    print('- - - - - - - - -')

    print(f"try to juxtapose with 3, 0")
    tmp = spb.juxtapose(arr3, 3, 0)
    print(f"juxtapose --->\n{tmp}")

    print('- - - - - - - - -')


    print(f"try to juxtapose with 1, 1")
    tmp = spb.juxtapose(arr3, 1, 1)
    print(f"juxtapose --->\n{tmp}")

    print('- - - - - - - - -')

    print(f"try to juxtapose with 0, 0")
    tmp = spb.juxtapose(arr3, 0, 0)
    print(f"juxtapose --->\n{tmp}")

    print('-------------------------------------------------------------------')



##################################################################



#################### MOSAIC #########################################


    arr4 = np.array([[0,1,2,3],[3,2,1,0],[1,1,1,1]])
    print(f"arr4 -->\n{arr4}")
    print(f"try to mosaic with 2, 2")
    tmp = spb.mosaic(arr4, (2,2))
    print(f"mosaic --->\n{tmp}")

    print('- - - - - - - - -')

    print(f"try to mosaic with 1, 1")
    tmp = spb.mosaic(arr4, (1, 1))
    print(f"mosaic --->\n{tmp}")

    print('- - - - - - - - -')

    print(f"try to mosaic with 0, 0")
    tmp = spb.mosaic(arr4, (0,0))
    print(f"mosaic --->\n{tmp}")

##################################################################
