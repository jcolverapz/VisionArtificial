Mat img = imread(argv[1]);

if(!src.data)
    cerr << "Problem loading image!!!" << endl;

imshow("img .jpg", img);

cvtColor(img, gray, CV_BGR2GRAY);
imshow("gray", gray);


Mat binary_image;
adaptiveThreshold(gray, binary_image, 255, CV_ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, -2);
imshow("binary.jpg", binary_image);

// Create the images that will use to extract the horizontal and vertical lines
Mat horizontal = binary_image.clone();
Mat vertical = binary_image.clone();

int horizontalsize = horizontal.cols / 30;

Mat horizontalStructure = getStructuringElement(MORPH_RECT, Size(horizontalsize,1));


erode(horizontal, horizontal, horizontalStructure, Point(-1, -1));
dilate(horizontal, horizontal, horizontalStructure, Point(-1, -1));
imshow("horizontal", horizontal);

int verticalsize = vertical.rows / 30;

Mat verticalStructure = getStructuringElement(MORPH_RECT, Size( 1,verticalsize));

erode(vertical, vertical, verticalStructure, Point(-1, -1));
dilate(vertical, vertical, verticalStructure, Point(-1, -1));

imshow("vertical", vertical);

bitwise_not(vertical, vertical);
imshow("vertical_bit", vertical);


Mat edges;
adaptiveThreshold(vertical, edges, 255, CV_ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 3, -2);
imshow("edges", edges);

Mat kernel = Mat::ones(2, 2, CV_8UC1);
dilate(edges, edges, kernel);
imshow("dilate", edges);

Mat smooth;
vertical.copyTo(smooth);

blur(smooth, smooth, Size(2, 2));

smooth.copyTo(vertical, edges);

imshow("smooth", vertical);
waitKey(0);
return 0;