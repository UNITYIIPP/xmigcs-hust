# setup.py
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy
import os
import sys
from setuptools.command.build_py import build_py


def get_cython_extensions():
    """获取所有需要编译的Python文件并创建Cython扩展"""
    source_dir = "src/xmigcs"
    extensions = []
    
    # 遍历源码目录，找到所有.py文件
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.py'):
                py_file = os.path.join(root, file)
                
                # 跳过 __init__.py 文件
                if file == '__init__.py':
                    continue
                
                # 计算模块名称
                rel_path = os.path.relpath(py_file, source_dir)
                module_name = rel_path.replace(os.sep, '.').replace('.py', '')
                
                # 检查文件内容以确定依赖
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().lower()
                
                include_dirs = []
                macros = []
                
                if 'numpy' in content:
                    include_dirs.append(numpy.get_include())
                    macros.append(("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"))
                
                # 创建扩展
                ext = Extension(
                    f"xmigcs.{module_name}",
                    [py_file],
                    include_dirs=include_dirs,
                    define_macros=macros,
                    extra_compile_args=["-O3", "-std=c99"],
                    extra_link_args=["-O3"],
                )
                extensions.append(ext)
    
    return extensions


class CustomBuildPy(build_py):
    """自定义 build_py，在非开发模式下排除 .py 文件"""
    
    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        
        # 如果是开发模式，保留所有文件
        if os.getenv('XMIGCS_DEV') == '1':
            return modules
        
        # 非开发模式：只保留 __init__.py 文件
        filtered_modules = []
        for pkg, mod, filename in modules:
            if mod == '__init__' or mod.endswith('__init__'):
                filtered_modules.append((pkg, mod, filename))
        return filtered_modules


# 检查是否是开发模式
is_dev_mode = os.getenv('XMIGCS_DEV') == '1'

if is_dev_mode:
    # 开发模式：安装源码
    setup(
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        zip_safe=False,
        cmdclass={
            'build_py': CustomBuildPy,
        },
        include_package_data=True,
    )
else:
    # 生产模式：编译 Cython 扩展
    extensions = get_cython_extensions()
    
    # 编译指令
    compiler_directives = {
        "language_level": 3,
        "boundscheck": False,
        "wraparound": False,
        "initializedcheck": False,
        "nonecheck": False,
        "cdivision": True,
        "embedsignature": True,
    }

    setup(
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        ext_modules=cythonize(
            extensions,
            compiler_directives=compiler_directives,
            annotate=False,
            language_level=3,
            nthreads=os.cpu_count(),
            build_dir="build/cython",
        ),
        cmdclass={
            'build_py': CustomBuildPy,
        },
        # 包含所有配置文件和 .so 文件
        package_data={
            'xmigcs': [
                '**/*.so',
                '**/*.yaml',
                '**/*.yml',
                '**/*.json',
                '**/*.onnx',
                '**/*.npz',
                '**/*.ipynb',
                '**/__init__.py',
                '**/*.urdf',
            ],
            'xmigcs.policy.supermimic': [
                'config/*.yaml', 'model/*.onnx', 'model/*.json', 'motion/*.npz',
            ],
            'xmigcs.policy.teleop_yibo': [
                'config/*.yaml', 'model/*.onnx', 'model/**/*.onnx',
                'model/*.npz', 'model/**/*.npz', 'motion_50hz/*.npz',
            ],
            'xmigcs.policy.whole_body_control': [
                'config/*.yaml', 'model/*.onnx', 'model/**/*.onnx',
                'model/*.npz', 'model/**/*.npz', 'motion_50hz/*.npz',
            ],
            'xmigcs.policy.allmimic': [
                'config/*.yaml', 'model/*.onnx', 'model/*.json', 'model/**/*.onnx', 'model/**/*.json',
            ],
        },
        # 排除其他 .py 文件
        exclude_package_data={
            'xmigcs': ['**/*.py'],
        },
        include_package_data=True,
        zip_safe=False,
    )
