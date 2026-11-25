// End-to-end test with a generated mixed-policy bundle: pass its directory as argv[1].
#include "deploy_onnx.h"
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace XBot::policy;

static void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

int main(int argc, char** argv)
{
    try {
        require(argc == 2, "usage: mixed_policy_runtime_test BUNDLE_DIR");
        const std::string dir(argv[1]);
        const auto md = YAML::LoadFile(dir + "/deploy_metadata.yaml");
        RobotInfo robot{};
        robot.joint_names = md["joint_names"].as<std::vector<std::string>>();
        std::reverse(robot.joint_names.begin(), robot.joint_names.end());
        const auto n = robot.joint_names.size();
        robot.base_T_imu = Eigen::Affine3d::Identity();
        robot.joint_pos_min = Eigen::VectorXf::Constant(n, -1000);
        robot.joint_pos_max = Eigen::VectorXf::Constant(n, 1000);
        robot.joint_vel_max = Eigen::VectorXf::Constant(n, 100000);
        OnnxPolicy policy(dir + "/policy.onnx", dir + "/deploy_metadata.yaml", "", robot);
        const auto info = policy.policyInfo();
        require(policy.is_mixed_mode() && info.mixed_modes.size() == 4 && info.action_size == 12, "wrong policy metadata");
        Inputs in;
        Outputs out(info.action_size, n);
        in.w_R_imu = Eigen::Quaterniond::Identity();
        in.imu_omega = Eigen::Vector3d(.11, -.17, .23);
        in.imu_acc = Eigen::Vector3d::Zero();
        in.command = policy.default_commands();
        in.command.at("base_velocity") << .27, -.09, .12;
        in.q = Eigen::VectorXd::Zero(n);
        in.v = in.tau = in.q_ref = in.v_ref = in.tau_ref = in.k = in.d = Eigen::VectorXd::Zero(n);
        // Seed an arbitrary, asymmetric last physically commanded target.
        for(std::size_t j = 0; j < n; ++j) {
            int r = info.joint_id_policy_to_robot[j];
            in.q[r] = .03 * (static_cast<int>(j) - 5);
            in.v[r] = .02 * (static_cast<int>(j) - 5);
            out.q_des[r] = -.04 * (static_cast<int>(j) - 5);
        }
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "mixed_policy_test");
        Ort::SessionOptions options;
        Ort::Session direct(env, (dir + "/policy.onnx").c_str(), options);
        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        for(const auto& mode : info.mixed_modes) {
            in.mode_selectors = mode.selectors;
            const Eigen::VectorXd previous_target = out.q_des;
            require(policy.run(in, out), "inference returned false");
            require(out.raw_observation.size() == 47 && out.raw_action.size() == 12, "invalid actor dimensions");
            for(std::size_t j = 0; j < n; ++j) {
                int r = info.joint_id_policy_to_robot[j];
                require(std::abs(out.raw_observation[9+j] - (in.q[r] - mode.joint_reference_rad[j])) < 1e-5, "wrong relative joint pose");
                require(std::abs(out.raw_observation[21+j] - in.v[r]) < 1e-5, "wrong joint velocity");
                require(std::abs(out.raw_observation[33+j] - (previous_target[r] - mode.joint_reference_rad[j]) / info.mixed_action_scale) < 1e-5,
                        "wrong applied-action rebase");
                require(std::abs(out.q_des[r] - (mode.joint_reference_rad[j] + info.mixed_action_scale*out.raw_action[j])) < 1e-5,
                        "wrong action target or joint mapping");
            }
            require(out.raw_observation[45] == mode.selectors[0] && out.raw_observation[46] == mode.selectors[1], "wrong selectors");
            std::array<int64_t, 2> shape{1, 47};
            auto input = Ort::Value::CreateTensor<float>(mem, out.raw_observation.data(), 47, shape.data(), shape.size());
            const char* input_name = "obs";
            const char* output_name = "actions";
            auto direct_output = direct.Run(Ort::RunOptions{nullptr}, &input_name, &input, 1, &output_name, 1);
            const float* raw = direct_output[0].GetTensorData<float>();
            for(int i = 0; i < 12; ++i)
                require(std::abs(raw[i] - out.raw_action[i]) < 1e-6, "runtime output differs from direct ONNX inference");
        }
        in.mode_selectors << 0.0, 1.0;
        bool rejected = false;
        try { policy.run(in, out); } catch(const std::runtime_error&) { rejected = true; }
        require(rejected, "invalid selectors were accepted");
        // Swap two otherwise unique pairs: name-based and selector-based
        // clients must never disagree about which physical mode is active.
        auto invalid_metadata = YAML::Clone(md);
        invalid_metadata["mixed_modes"]["bb"]["selectors"][0] = 1.0;
        invalid_metadata["mixed_modes"]["fb"]["selectors"][0] = -1.0;
        auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        const auto temp = std::filesystem::temp_directory_path() / ("kyon_bad_modes_" + std::to_string(stamp) + ".yaml");
        { std::ofstream file(temp); file << invalid_metadata; require(bool(file), "cannot write malformed fixture"); }
        rejected = false;
        try { OnnxPolicy bad(dir + "/policy.onnx", temp.string(), "", robot); }
        catch(const std::runtime_error&) { rejected = true; }
        std::filesystem::remove(temp);
        require(rejected, "mismatched mode name and selectors were accepted");
        std::cout << "All four modes, previous-target rebasing, reordered joint mapping and selector validation passed.\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "Mixed runtime test failed: " << e.what() << '\n';
        return 1;
    }
}
