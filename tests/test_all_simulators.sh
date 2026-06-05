#!/bin/bash

# Test All Simulators Script
# 一键测试3个仿真器（IsaacGym, Genesis, IsaacLab）中所有任务的可执行性

# 配置
CONDA_ENV_GYM="lr_gym"
CONDA_ENV_GEN="lr_gen"
CONDA_ENV_LAB="lr_lab"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 检测conda环境是否存在
check_conda_env() {
    local env_name=$1
    conda env list 2>/dev/null | grep -q "^${env_name}\s"
    return $?
}

# 在指定conda环境中运行测试
run_test_in_env() {
    local env_name=$1
    local simulator_name=$2
    local extra_exports=$3
    
    echo -e "${BLUE}==========================================${NC}"
    echo -e "${BLUE}  正在测试: $simulator_name${NC}"
    echo -e "${BLUE}  环境: $env_name${NC}"
    echo -e "${BLUE}==========================================${NC}"
    echo ""
    
    # 使用bash -c 创建子shell来运行测试
    # 这样可以隔离每个仿真器的环境
    bash -c "
        # 初始化conda - 使用更可靠的方式
        if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
            source ~/miniconda3/etc/profile.d/conda.sh
        elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
            source ~/anaconda3/etc/profile.d/conda.sh
        elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
            source /opt/conda/etc/profile.d/conda.sh
        else
            # 备选方案：尝试eval方式
            eval \"\$(conda shell.bash hook)\"
        fi
        
        # 激活环境
        conda activate $env_name
        if [ \$? -ne 0 ]; then
            echo '错误: 无法激活环境 $env_name'
            exit 1
        fi
        
        # 设置额外的环境变量
        $extra_exports
        
        echo 'Python: '\$(which python)'
        echo 'Python版本: '\$(python --version)'
        echo '仿真器: $simulator_name'
        echo ''
        
        # 运行测试
        cd $PROJECT_DIR
        python $SCRIPT_DIR/test_all_tasks.py --headless
        exit \$?
    "
    
    return $?
}

# 主函数
main() {
    echo -e "${GREEN}==========================================${NC}"
    echo -e "${GREEN}    多仿真器测试脚本${NC}"
    echo -e "${GREEN}==========================================${NC}"
    echo ""
    
    # 解析参数
    local iterations=5
    local test_tasks=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --iterations)
                iterations="$2"
                shift 2
                ;;
            --tasks)
                shift
                test_tasks="$1"
                shift
                ;;
            --help)
                echo "用法: $0 [选项]"
                echo ""
                echo "选项:"
                echo "  --iterations N    每个任务运行N次迭代 (默认: 5)"
                echo "  --tasks TASKS     指定要测试的任务，用逗号分隔"
                echo "  --help            显示此帮助信息"
                echo ""
                echo "示例:"
                echo "  $0                                  # 测试所有仿真器的所有任务"
                echo "  $0 --iterations 3                   # 每个任务运行3次迭代"
                echo "  $0 --tasks go2,go2_stage2           # 只测试指定Go2任务"
                exit 0
                ;;
            *)
                echo "未知选项: $1"
                echo "使用 --help 查看帮助"
                exit 1
                ;;
        esac
    done
    
    # 检测可用的环境
    echo "检测conda环境..."
    local has_gym=false
    local has_gen=false
    local has_lab=false
    
    if check_conda_env "$CONDA_ENV_GYM"; then
        echo -e "  [${GREEN}✓${NC}] $CONDA_ENV_GYM (IsaacGym)"
        has_gym=true
    else
        echo -e "  [${RED}✗${NC}] $CONDA_ENV_GYM (IsaacGym) - 跳过"
    fi
    
    if check_conda_env "$CONDA_ENV_GEN"; then
        echo -e "  [${GREEN}✓${NC}] $CONDA_ENV_GEN (Genesis)"
        has_gen=true
    else
        echo -e "  [${RED}✗${NC}] $CONDA_ENV_GEN (Genesis) - 跳过"
    fi
    
    if check_conda_env "$CONDA_ENV_LAB"; then
        echo -e "  [${GREEN}✓${NC}] $CONDA_ENV_LAB (IsaacLab)"
        has_lab=true
    else
        echo -e "  [${RED}✗${NC}] $CONDA_ENV_LAB (IsaacLab) - 跳过"
    fi
    
    echo ""
    
    # 检查结果
    local results=()
    local total_passed=0
    local total_failed=0
    
    # 构建测试参数
    local test_args="--iterations $iterations"
    if [ -n "$test_tasks" ]; then
        test_args="$test_args --tasks $test_tasks"
    fi
    
    # 测试IsaacGym
    if [ "$has_gym" = true ]; then
        echo -e "${YELLOW}>>> 开始测试 IsaacGym${NC}"
        run_test_in_env "$CONDA_ENV_GYM" "IsaacGym" "export LD_LIBRARY_PATH=/home/lupinjia/miniconda3/envs/lr_gym/lib:\$LD_LIBRARY_PATH"
        local gym_result=$?
        results+=("IsaacGym:$gym_result")
        if [ $gym_result -eq 0 ]; then
            ((total_passed++))
        else
            ((total_failed++))
        fi
        echo ""
        echo ""
    fi
    
    # 测试Genesis
    if [ "$has_gen" = true ]; then
        echo -e "${YELLOW}>>> 开始测试 Genesis${NC}"
        run_test_in_env "$CONDA_ENV_GEN" "Genesis" "export SIMULATOR=genesis"
        local gen_result=$?
        results+=("Genesis:$gen_result")
        if [ $gen_result -eq 0 ]; then
            ((total_passed++))
        else
            ((total_failed++))
        fi
        echo ""
        echo ""
    fi
    
    # 测试IsaacLab
    if [ "$has_lab" = true ]; then
        echo -e "${YELLOW}>>> 开始测试 IsaacLab${NC}"
        run_test_in_env "$CONDA_ENV_LAB" "IsaacLab" "export SIMULATOR=isaaclab"
        local lab_result=$?
        results+=("IsaacLab:$lab_result")
        if [ $lab_result -eq 0 ]; then
            ((total_passed++))
        else
            ((total_failed++))
        fi
        echo ""
        echo ""
    fi
    
    # 打印总结
    echo -e "${GREEN}==========================================${NC}"
    echo -e "${GREEN}           测试结果汇总${NC}"
    echo -e "${GREEN}==========================================${NC}"
    echo ""
    
    for result in "${results[@]}"; do
        IFS=':' read -r sim status <<< "$result"
        if [ "$status" -eq 0 ]; then
            echo -e "  [${GREEN}✓${NC}] $sim: 通过"
        else
            echo -e "  [${RED}✗${NC}] $sim: 失败"
        fi
    done
    
    echo ""
    echo "总计:"
    echo -e "  通过: ${GREEN}$total_passed${NC}"
    echo -e "  失败: ${RED}$total_failed${NC}"
    
    if [ $total_failed -eq 0 ] && [ $total_passed -gt 0 ]; then
        echo ""
        echo -e "${GREEN}🎉 所有仿真器测试通过！${NC}"
        exit 0
    elif [ $total_passed -eq 0 ] && [ $total_failed -eq 0 ]; then
        echo ""
        echo -e "${YELLOW}⚠️  没有可用的仿真器环境${NC}"
        exit 1
    else
        echo ""
        echo -e "${RED}⚠️  部分仿真器测试失败${NC}"
        exit 1
    fi
}

# 执行主函数
main "$@"
