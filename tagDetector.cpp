#include <iostream>
#include <atomic>
#include <thread>
#include <cstdio>

#include <opencv4/opencv2/opencv.hpp>

#include "apriltag.h"
#include "tag36h11.h"
#include "apriltag_pose.h"

std::atomic<bool> stop_main;

void check_for_input() {
    while (true) {
        std::string input;
        std::cin >> input;
        if (input == "q") {
            stop_main = true;
            break;
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: ./tagDetector <output_file>\n";
        return -1;
    }
    
    std::thread input_thread(check_for_input);

    FILE* outfile = fopen(argv[1], "w+");
    if (outfile == nullptr) {
        perror("Failed to create output file\n");
    }

    cv::VideoCapture cap(4);
    if (!cap.isOpened()) {
        std::cerr << "Couldn't open video capture device\n";
        return -1;
    }

    apriltag_family_t *tf = tag36h11_create();
    apriltag_detector_t *td = apriltag_detector_create();   
    apriltag_detector_add_family(td, tf);

    auto start_time = std::chrono::high_resolution_clock::now();
    bool recording_started = false;
    long missed_samples = 0;

    while (true) {
        if (stop_main) {
            break;
        }

        cv::Mat frame, gray;
        cap >> frame;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        image_u8_t im = { .width = gray.cols,
            .height = gray.rows,
            .stride = gray.cols,
            .buf = gray.data
        };

        zarray_t *detections = apriltag_detector_detect(td, &im);
        
        if (zarray_size(detections) == 0) {
            if (recording_started) {
                missed_samples++;
            }
            cv::imshow("Video", frame);
            if (cv::waitKey(30) >= 0) {
                break;
            }
            continue;
        }

        if (zarray_size(detections) > 1) {
            std::cerr << "Too many april tags in image!\n";
            return -1;
        }
        recording_started = true;

        apriltag_detection* det;
        zarray_get(detections, 0, &det);

        cv::line(frame, cv::Point(det->p[0][0], det->p[0][1]),
                     cv::Point(det->p[1][0], det->p[1][1]),
                     cv::Scalar(0, 0xff, 0), 2);
        cv::line(frame, cv::Point(det->p[0][0], det->p[0][1]),
                    cv::Point(det->p[3][0], det->p[3][1]),
                    cv::Scalar(0, 0, 0xff), 2);
        cv::line(frame, cv::Point(det->p[1][0], det->p[1][1]),
                    cv::Point(det->p[2][0], det->p[2][1]),
                    cv::Scalar(0xff, 0, 0), 2);
        cv::line(frame, cv::Point(det->p[2][0], det->p[2][1]),
                    cv::Point(det->p[3][0], det->p[3][1]),
                    cv::Scalar(0xff, 0, 0), 2);

        cv::imshow("Video", frame);
        if (cv::waitKey(30) >= 0) {
            break;
        }

        // First create an apriltag_detection_info_t struct using your known parameters.
        apriltag_detection_info_t info;
        info.det = det;
        info.tagsize = 0.1;
        info.fx = 442.0382;
        info.fy = 441.7077;
        info.cx = 463.1482;
        info.cy = 275.9004;

        apriltag_pose_t pose;
        estimate_tag_pose(&info, &pose);

        std::string output;
        auto sample_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(sample_time - start_time);
        output += std::to_string(duration.count()) + " ";

        std::cout << "Rotation:\n";
        for (int i = 0; i < pose.R->nrows; i++) {
            for (int j = 0; j < pose.R->ncols; j++) {
                std::cout << pose.R->data[i * pose.R->ncols + j] << " ";
                output += std::to_string(pose.R->data[i * pose.R->ncols + j]) + " ";
            }
            output += std::to_string(pose.t->data[i]) + " ";
            std::cout << "\n";
        }
        output += "0 0 0 1\n";
        fwrite(output.data(), output.size(), 1, outfile);

        std::cout << "T\n";
        for (int i = 0; i < pose.t->nrows; i++) {
            for (int j = 0; j < pose.t->ncols; j++) {
                std::cout << pose.t->data[i * pose.t->ncols + j] << " ";
            }
            std::cout << "\n";
        }

        apriltag_detections_destroy(detections);
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
    }
    std::cout << "missed samples: " << missed_samples << "\n";

    fclose(outfile);

    apriltag_detector_destroy(td);
    tag36h11_destroy(tf);

    input_thread.join();

    return 0;
}
