from PIL import Image, ImageDraw
import struct
import math;


def heatmap(minimum, maximum, value):
	if (value < 0 ): value = 0;
	minimum, maximum = float(minimum), float(maximum)
	ratio = 2 * (value-minimum) / (maximum - minimum)
	r = min(255,int(255*ratio))
	g = min(255,int(255*ratio))
	b = min(255,int(255*ratio))
	return r, g, b


inf = open("output.dat", "rb");
widthr = inf.read(4);
width = struct.unpack("i", widthr)[0];
heightr = inf.read(4);
height = struct.unpack("i", heightr)[0];

print( "" + str(width) + ", " + str(height) );

img = Image.new("RGB", (int(width/4),int(height/4)), "white");
pixels = img.load()

maxv = 0;
minv = 100000;
state = [];
for idx in range (width*height):
	r = inf.read(4);
	f = struct.unpack("f", r);
	fs = 0;
	if ( math.isfinite(f[0]) ): fs = f[0];
	#else: print(" ...");

	state.append(fs);
	if ( maxv < fs ): maxv = fs;
	if ( minv > fs ): minv = fs;

print ("maxv " + str(maxv) + ", minv " + str(minv));
for x in range (width):
	for y in range (height):
		idx = int(y*width + x);
		#if ( fs > 255 ): fs = 255;
		if ( state[idx] < 0 ): state[idx] = 0;
		#pixels[x,y] = (fs,fs,fs);
		pixels[x/4,y/4] = heatmap(0,0.15,state[idx]);
		# if ( state[idx] == 500 ): print(str(y)+"\t"+str(x)+"\n");


#for idx in range (width*height):
#	fs = int(state[idx] * (255/maxv));
#	if ( fs > 255 ): fs = 255;
#	x = idx%width;
#	y = idx/width;
#	
#	if ( fs > 255 ): fs = 255;
#	if ( fs < 0 ): fs = 0;
#	pixels[x,y] = (fs,fs,fs);

img.save("render.png");





