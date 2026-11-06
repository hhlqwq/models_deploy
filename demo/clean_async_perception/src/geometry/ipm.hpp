// IPM (Inverse Perspective Mapping)：将图像 bbox 映射到地面物理坐标。
#pragma once

#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <map>
#include <iostream>

struct PhysicalBBox {
    float width_cm = -1.0f;
    float height_cm = -1.0f;
    float area_cm2 = -1.0f;
    bool valid = false;
};

struct IpmConfig {
    float camera_height_m = 0.75f;  // 相机离地面高度 (米)
    float pitch_deg = 45.0f;        // 相机俯仰角 (度)，正值=向下倾斜
    float roll_deg = 0.0f;          // 相机横滚角 (度)
    float yaw_deg = 0.0f;           // 相机偏航角 (度)
    // 手动内参覆盖（SDK 无法获取时使用）
    float manual_fx = 0.0f;
    float manual_fy = 0.0f;
    float manual_cx = 0.0f;
    float manual_cy = 0.0f;
    int manual_img_w = 0;
    int manual_img_h = 0;
};

class IpmCalculator {
   public:
    IpmCalculator() = default;

    // 从 INI 配置和相机内参初始化
    bool init(const IpmConfig& cfg, float fx, float fy, float cx, float cy,
              int imgWidth, int imgHeight) {
        fx_ = fx;
        fy_ = fy;
        cx_ = cx;
        cy_ = cy;
        imgW_ = imgWidth;
        imgH_ = imgHeight;
        height_ = cfg.camera_height_m;

        const float pitchRad = cfg.pitch_deg * 3.14159265f / 180.0f;
        const float rollRad = cfg.roll_deg * 3.14159265f / 180.0f;
        const float yawRad = cfg.yaw_deg * 3.14159265f / 180.0f;

        // 构建相机到世界的旋转矩阵
        // 相机坐标系: Xc=右, Yc=下, Zc=前(光轴)
        // 世界坐标系: Xw=右, Yw=前, Zw=上
        // 相机朝下 pitch θ 角时 (无 yaw/roll):
        //   Xc → (1, 0, 0)         水平右
        //   Yc → (0, -sinθ, -cosθ) 相机"下" → 世界"后下方"
        //   Zc → (0, cosθ, -sinθ)  相机"前" → 世界"前下方"
        float cp = std::cos(pitchRad), sp = std::sin(pitchRad);
        float cyaw = std::cos(yawRad), syaw = std::sin(yawRad);
        float cr = std::cos(rollRad), sr = std::sin(rollRad);

        // 基础 pitch 旋转: Rpitch = [Xc|Yc|Zc]
        float Rpitch[9] = {1.0f, 0.0f, 0.0f, 0.0f, -sp, cp, 0.0f, -cp, -sp};
        // Yaw: 绕世界 Z 轴
        float Ry[9] = {cyaw, -syaw, 0.0f, syaw, cyaw, 0.0f, 0.0f, 0.0f, 1.0f};
        // Roll: 绕相机光轴 Zc
        float Rr[9] = {cr, -sr, 0.0f, sr, cr, 0.0f, 0.0f, 0.0f, 1.0f};
        // R_total = Ry * Rpitch * Rr
        float tmp[9];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                tmp[i * 3 + j] = 0;
                for (int k = 0; k < 3; ++k)
                    tmp[i * 3 + j] += Rpitch[i * 3 + k] * Rr[k * 3 + j];
            }
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                R_[i * 3 + j] = 0;
                for (int k = 0; k < 3; ++k)
                    R_[i * 3 + j] += Ry[i * 3 + k] * tmp[k * 3 + j];
            }

        initialized_ = true;
        return true;
    }

    bool isInitialized() const { return initialized_; }

    // 图像像素 → 地面坐标 (米)
    bool imageToGround(float u, float v, float* X, float* Y) const {
        if (!initialized_) return false;
        // 归一化图像坐标
        const float x_n = (u - cx_) / fx_;
        const float y_n = (v - cy_) / fy_;
        // 相机射线方向 (相机坐标系)
        const float d[3] = {x_n, y_n, 1.0f};
        // 旋转到世界坐标系
        const float dw[3] = {R_[0] * d[0] + R_[1] * d[1] + R_[2] * d[2],
                             R_[3] * d[0] + R_[4] * d[1] + R_[5] * d[2],
                             R_[6] * d[0] + R_[7] * d[1] + R_[8] * d[2]};
        // 射线与地面 (Z_w = 0) 求交
        // 相机在世界坐标系位置: (0, 0, height)
        // P = (0, 0, height) + t * dw
        // Z_w = height + t * dw[2] = 0  =>  t = -height / dw[2]
        if (std::abs(dw[2]) < 1e-6f) return false;  // 射线平行于地面
        const float t = -height_ / dw[2];
        if (t < 0) return false;  // 地面在相机后方
        *X = t * dw[0];           // 世界 X (右)
        *Y = t * dw[1];           // 世界 Y (前)
        return true;
    }

    // 计算 bbox 的物理面积
    PhysicalBBox computePhysicalBBox(float x1, float y1, float x2,
                                     float y2) const {
        PhysicalBBox result;
        if (!initialized_) return result;

        float X_tl, Y_tl, X_tr, Y_tr, X_bl, Y_bl, X_br, Y_br;
        // 四个角点: 左上, 右上, 左下, 右下
        bool ok1 = imageToGround(x1, y1, &X_tl, &Y_tl);
        bool ok2 = imageToGround(x2, y1, &X_tr, &Y_tr);
        bool ok3 = imageToGround(x1, y2, &X_bl, &Y_bl);
        bool ok4 = imageToGround(x2, y2, &X_br, &Y_br);

        if (!ok1 || !ok2 || !ok3 || !ok4) return result;

        // 地面上的宽度 (X方向) 和高度 (Y方向)
        // 取上下边的平均宽度
        const float topWidth = std::abs(X_tr - X_tl);
        const float botWidth = std::abs(X_br - X_bl);
        const float avgWidth = (topWidth + botWidth) / 2.0f;

        // 取左右边的平均高度
        const float leftHeight = std::abs(Y_bl - Y_tl);
        const float rightHeight = std::abs(Y_br - Y_tr);
        const float avgHeight = (leftHeight + rightHeight) / 2.0f;

        result.width_cm = avgWidth * 100.0f;    // m → cm
        result.height_cm = avgHeight * 100.0f;  // m → cm
        result.area_cm2 = result.width_cm * result.height_cm;
        result.valid = true;
        return result;
    }

   private:
    bool initialized_ = false;
    float fx_ = 0, fy_ = 0, cx_ = 0, cy_ = 0;
    int imgW_ = 0, imgH_ = 0;
    float height_ = 0.75f;
    float R_[9] = {};  // 3x3 旋转矩阵 (相机→世界)
};

// 从 INI map 加载 IPM 参数
inline IpmConfig loadIpmConfig(
    const std::map<std::string, std::string>& values) {
    IpmConfig cfg;
    auto getFloat = [&](const std::string& key, float def) -> float {
        auto it = values.find(key);
        if (it == values.end()) return def;
        try {
            return std::stof(it->second);
        } catch (...) {
            return def;
        }
    };
    cfg.camera_height_m = getFloat("ipm.camera_height_m", 0.75f);
    cfg.pitch_deg = getFloat("ipm.pitch_deg", 45.0f);
    cfg.roll_deg = getFloat("ipm.roll_deg", 0.0f);
    cfg.yaw_deg = getFloat("ipm.yaw_deg", 0.0f);
    cfg.manual_fx = getFloat("ipm.fx", 0.0f);
    cfg.manual_fy = getFloat("ipm.fy", 0.0f);
    cfg.manual_cx = getFloat("ipm.cx", 0.0f);
    cfg.manual_cy = getFloat("ipm.cy", 0.0f);
    cfg.manual_img_w = static_cast<int>(getFloat("ipm.img_width", 0.0f));
    cfg.manual_img_h = static_cast<int>(getFloat("ipm.img_height", 0.0f));
    return cfg;
}
