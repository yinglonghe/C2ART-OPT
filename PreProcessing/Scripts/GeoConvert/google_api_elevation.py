from json import loads
from time import sleep
from urllib.request import Request, urlopen
 
# locations=[(46.89634563, 16.84353007), (46.89626888, 16.84287862)] #(lat,lon) pairs

def LatLon2Alt(locations):
    alt = []
    success = 0
    while success == 0:
        success = 1
        for i, loc in enumerate(locations): 
            try:
                request = Request('https://maps.googleapis.com/maps/api/elevation/json?locations={0},{1}&key=AIzaSyDpraSS8zU7mNG2Jwzru_Xere537tsBRtc'.format(loc[0],loc[1]))
                response = urlopen(request).read() 
                places = loads(response)
                alt.append(places['results'][0]['elevation'])
                # print('At {0} elevation is: {1}'.format(loc, places['results'][0]['elevation']))
                # sleep(1)
                if i % 100 == 0:
                    print(str(i) + '/' + str(len(locations)))
            except:
                print('Error for location: {0}'.format(loc))
                success = 0
                break

    return alt