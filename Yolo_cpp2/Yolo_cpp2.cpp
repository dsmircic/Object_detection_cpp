// Include Libraries.
#include <opencv2/opencv.hpp>
#include <fstream>
#include <tuple>
#include "config.hpp"

#define ROWS 25200
#define COLUMNS_TO_SKIP 5
#define SKIP_FRAMES 10

// rotate image 
cv::Mat RotateImage(cv::Mat& input_frame)
{
    double angle = 90;

    // get the center coordinates of the image to create the 2D rotation matrix
    cv::Point2f center((input_frame.cols - 1) / 2.0, (input_frame.rows - 1) / 2.0);
    // using getRotationMatrix2D() to get the rotation matrix
    cv::Mat rotation_matix = cv::getRotationMatrix2D(center, angle, 1.0);

    // we will save the resulting image in rotated_image matrix
    cv::Mat rotated_frame;
    // rotate the image using warpAffine
    warpAffine(input_frame, rotated_frame, rotation_matix, input_frame.size());
    return rotated_frame;
}
// Draw the predicted bounding box.
void drawLabel(cv::Mat& input_image, std::string label, int left, int top)
{
    // Display the label at the top of the bounding box.
    int baseLine;
    cv::Size label_size = cv::getTextSize(label, font::FONT_FACE, font::FONT_SCALE, font::THICKNESS, &baseLine);
    top = cv::max(top, label_size.height);

    // Top left corner.
    cv::Point tlc = cv::Point(left, top);
    // Bottom right corner.
    cv::Point brc = cv::Point(left + label_size.width, top + label_size.height + baseLine);
    // Draw black rectangle.
    rectangle(input_image, tlc, brc, color::BLACK, cv::FILLED);
    // Put the label on the black rectangle.
    putText(input_image, label, cv::Point(left, top + label_size.height), font::FONT_FACE, font::FONT_SCALE, color::YELLOW, font::THICKNESS);
}


std::vector<cv::Mat> preProcess(cv::Mat& input_image, cv::dnn::Net& net)
{
    // Convert to blob.
    cv::Mat blob;
    cv::dnn::blobFromImage(input_image, blob, 1. / 255., cv::Size(detection_constants::INPUT_WIDTH, detection_constants::INPUT_HEIGHT), cv::Scalar(), true, false);

    net.setInput(blob);

    // Forward propagate.
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    return outputs;
}

cv::Rect findBase(std::vector<int> classes, std::vector<cv::Rect> bboxes)
{
    cv::Rect base_bbox = cv::Rect(0, 0, 0, 0);

    for (int i = 0; i < classes.size(); i++)
    {
        if (classes[i] == base::base)
            base_bbox = bboxes[i];
    }

    return base_bbox;
}

bool checkObjectInBase(cv::Rect obj_bbox, cv::Rect base_bbox)
{

    std::cout << base_bbox << std::endl;

    if (base_bbox.area() == 0)
        return false;

    if (base::base == -1)
        return true;

    if (base_bbox.contains(obj_bbox.tl()) || base_bbox.contains(obj_bbox.br()))
        return true;

    return false;
}

std::tuple <cv::Mat, std::vector<cv::Rect>, std::vector<int>, std::vector<float>, std::vector<int>, std::vector<std::string>> post_process(cv::Mat& inputImage, std::vector<cv::Mat>& outputs, const std::vector<std::string>& class_name)
{
    // Initialize std::vectors to hold respective outputs while unwrapping detections.
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // Resizing factor.
    float x_factor = inputImage.cols / detection_constants::INPUT_WIDTH;
    float y_factor = inputImage.rows / detection_constants::INPUT_HEIGHT;

    float* data = (float*)outputs[0].data;

    const int dimensions = 9;

    // Iterate through 25200 detections.
    for (int i = 0; i < ROWS; ++i)
    {
        float confidence = data[4];
        // Discard bad detections and continue.
        if (confidence >= detection_constants::CONFIDENCE_THRESHOLD)
        {
            float* classes_scores = data + 5;
            // Create a 1x85 cv::Mat and store class scores of 80 classes.
            cv::Mat scores(1, class_name.size(), CV_32FC1, classes_scores);

            // Perform minMaxLoc and acquire index of best class score.
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

            // Continue if the class score is above the threshold.
            if (max_class_score > detection_constants::SCORE_THRESHOLD)
            {
                // Store class ID and confidence in the pre-defined respective std::vectors.

                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);

                // Center.
                float cx = data[0];
                float cy = data[1];

                // Box dimension.
                float w = data[2];
                float h = data[3];

                // Bounding box coordinates.
                int left = int((cx - 0.5 * w) * x_factor);
                int top = int((cy - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);

                // Store good detections in the boxes std::vector.
                boxes.push_back(cv::Rect(left, top, width, height));
            }

        }
        // Jump to the next column.
        data += 4 + COLUMNS_TO_SKIP;
    }

    // Perform Non Maximum Suppression and draw predictions.
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, detection_constants::SCORE_THRESHOLD, detection_constants::NMS_THRESHOLD, indices);

    cv::Rect baseBoundingBox = findBase(class_ids, boxes);
    for (int i = 0; i < indices.size(); i++)
    {
        int idx = indices[i];
        cv::Rect objBoundingBox = boxes[idx];

        bool draw = checkObjectInBase(objBoundingBox, baseBoundingBox);

        if (draw)
        {
            int left = objBoundingBox.x;
            int top = objBoundingBox.y;
            int width = objBoundingBox.width;
            int height = objBoundingBox.height;
            // Draw bounding box.
            rectangle(inputImage, cv::Point(left, top), cv::Point(left + width, top + height), color::BLUE, 3 * font::THICKNESS);

            // Get the label for the class name and its confidence.
            std::string label = cv::format("%.2f", confidences[idx]);
            label = class_name[class_ids[idx]] + ":" + label;
            // Draw class labels.
            drawLabel(inputImage, label, left, top);
        }

    }
    return { inputImage, boxes, indices, confidences , class_ids, class_name };
}
cv::Mat justDraw(cv::Mat inputImage, std::vector<cv::Rect>& boxes, std::vector<int>& indices, std::vector<float>& confidences, std::vector<int>& classIDs, std::vector<std::string>& className)
{
    cv::Rect base_bbox = findBase(classIDs, boxes);
    for (int i = 0; i < indices.size(); i++)
    {
        int idx = indices[i];
        cv::Rect obj_bbox = boxes[idx];

        bool draw = checkObjectInBase(obj_bbox, base_bbox);

        if (draw)
        {
            int left = obj_bbox.x;
            int top = obj_bbox.y;
            int width = obj_bbox.width;
            int height = obj_bbox.height;

            // Draw bounding box.
            rectangle(inputImage, cv::Point(left, top), cv::Point(left + width, top + height), color::BLUE, 3 * font::THICKNESS);

            // Get the label for the class name and its confidence.
            std::string label = cv::format("%.2f", confidences[idx]);
            label = className[classIDs[idx]] + ":" + label;

            // Draw class labels.
            drawLabel(inputImage, label, left, top);
        }

    }
    return inputImage;
}

int main()
{
    // Load class list.
    std::vector<std::string> class_list;
    std::ifstream ifs("C:/Users/a889467/YOLO_config/reduced_classes.txt");
    std::string line;

    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }

    //// Load image.
    cv::Mat frame;
    //frame = imread("C:/Users/a879707/Documents/cpp/data/test.png");

    // Load model.
    cv::dnn::Net net;
    net = cv::dnn::readNet("C:/Users/a889467/YOLO_config/yolov5n_reduced.onnx");

    //namedWindow("Display window");

    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {

        std::cout << "cannot open camera";

    }
    //VideoWriter video("../../detect/outcpp.mp4", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, Size(INPUT_WIDTH, INPUT_HEIGHT));
    int numFrame = 0;
    bool rotate = false;

    cv::Mat img;
    std::vector<cv::Rect> boxes;
    std::vector<int> indices;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    std::vector<std::string> class_name;

    while (true) {
        numFrame++;
        cap >> frame;
        if (rotate)
        {
            frame = RotateImage(frame);
        }

        if (numFrame % SKIP_FRAMES == 0)
        {
            std::vector<cv::Mat> detections;
            detections = preProcess(frame, net);
            cv::Mat clone = frame.clone();
            tie(img, boxes, indices, confidences, class_ids, class_name) = post_process(clone, detections, class_list);
        }

        else
        {
            img = justDraw(frame, boxes, indices, confidences, class_ids, class_name);
        }

        std::vector<double> layersTimes;
        double freq = cv::getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        std::string label = cv::format("Inference time : %.2f ms", t);
        putText(img, label, cv::Point(20, 40), font::FONT_FACE, font::FONT_SCALE, color::RED);

        //video.write(img);
        cv::imshow("Output", img);
        cv::waitKey(25);

    }

    return 0;
}


