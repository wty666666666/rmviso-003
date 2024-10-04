#ifndef WINDMILL_H_
#define WINDMILL_H_

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>
#include <random>

namespace WINDMILL//定义空间
{
    class WindMill//定义类
    {


    private://私有变量
        int cnt;
        bool direct;
        double A;
        double w;
        double A0;//大概是角速度
        double fai;
        double now_angle;
        double start_time;
        cv::Point2i R_center;



        //5个私有函数
        void drawR(cv::Mat &img, const cv::Point2i &center);//绘制风车
        void drawHitFan(cv::Mat &img, const cv::Point2i &center, double angle);//绘制风车
        void drawOtherFan(cv::Mat &img, const cv::Point2i &center, double angle);//绘制风车




        cv::Point calPoint(const cv::Point2f &center, double angle_deg, double r)//计算并返回风车叶片上某点的坐标，三个输入参数分别是 中心点坐标，角度，半径。
                                                                                 //输出参数是计算出的新的坐标。
        {
            return center + cv::Point2f((float)cos(angle_deg / 180 * 3.1415926), (float)-sin(angle_deg / 180 * 3.1415926)) * r;
        }




        double SumAngle(double angle_now, double t0, double dt)//这个函数用于计算并更新风车叶片的角度
        {
            double dangle = A0 * dt + (A / w) * (cos(w * t0 + 1.81) - cos(w * (t0 + dt) + 1.81));
            angle_now += dangle / 3.1415926 * 180;
            if (angle_now < 0)
            {
                angle_now = 360 + angle_now;
            }
            if (angle_now > 360)
            {
                angle_now -= 360;
            }
            return angle_now;
        }

    public:
        WindMill(double time = 0);
        cv::Mat getMat(double time);

    };
} // namespace WINDMILL

#endif