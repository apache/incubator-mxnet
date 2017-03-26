// -*- mode: groovy -*-
// Jenkins pipeline
// See documents at https://jenkins.io/doc/book/pipeline/jenkinsfile/

// mxnet libraries
mx_lib = 'lib/libmxnet.so, lib/libmxnet.a, dmlc-core/libdmlc.a, nnvm/lib/libnnvm.a'
// command to start a docker container
docker_run = 'tests/ci_build/ci_build.sh'
// timeout in minutes
max_time = 60

// initialize source codes
def init_git() {
    checkout scm
    retry(5) {
        timeout(time: 2, unit: 'MINUTES') {
            sh 'git submodule update --init'
        }
    }
}

stage("Sanity Check") {
    timeout(time: max_time, unit: 'MINUTES') {
        node('linux') {
            ws('workspace/sanity') {
                init_git()
                make('lint', 'cpplint rcpplint jnilint')
                make('lint', 'pylint')
            }
        }
    }
}

// Run make. First try to do an incremental make from a previous workspace in hope to
// accelerate the compilation. If something wrong, clean the workspace and then
// build from scratch.
def make(docker_type, make_flag) {
    timeout(time: max_time, unit: 'MINUTES') {
        try {
            sh "${docker_run} ${docker_type} make ${make_flag}"
        } catch (exc) {
            echo 'Incremental compilation failed. Fall back to build from scratch'
            sh "${docker_run} ${docker_type} make clean"
            sh "${docker_run} ${docker_type} make ${make_flag}"
        }
    }
}

// pack libraries for later use
def pack_lib(name, libs=mx_lib) {
    sh """
echo "Packing ${libs} into ${name}"
echo ${libs} | sed -e 's/,/ /g' | xargs md5sum
"""
    stash includes: libs, name: name
}


// unpack libraries saved before
def unpack_lib(name, libs=mx_lib) {
    unstash name
    sh """
echo "Unpacked ${libs} from ${name}"
echo ${libs} | sed -e 's/,/ /g' | xargs md5sum
"""
}

stage('Build') {
    parallel 'CPU: Openblas': {
        node('linux') {
            ws('workspace/build-cpu') {
                init_git()
                def flag = """ \
USE_PROFILER=1                \
USE_BLAS=openblas             \
-j\$(nproc)
"""
                make("cpu", flag)
                pack_lib('cpu')
            }
        }
    },
            'GPU: CUDA7.5+cuDNN5': {
                node('GPU' && 'linux') {
                    ws('workspace/build-gpu') {
                        init_git()
                        def flag = """ \
USE_PROFILER=1                \
USE_BLAS=openblas             \
USE_CUDA=1                    \
USE_CUDA_PATH=/usr/local/cuda \
USE_CUDNN=1                   \
-j\$(nproc)
"""
                        make('gpu', flag)
                        pack_lib('gpu')
                    }
                }
            },
            'Amalgamation': {
                node('linux') {
                    ws('workspace/amalgamation') {
                        init_git()
                        make('cpu', '-C amalgamation/ USE_BLAS=openblas MIN=1')
                    }
                }
            },
            'CPU: MKLML': {
                node('linux') {
                    ws('workspace/build-mklml') {
                        init_git()
                        def flag = """ \
USE_PROFILER=1                \
USE_BLAS=openblas             \
USE_MKL2017=1                 \
USE_MKL2017_EXPERIMENTAL=1    \
USE_CUDA=1                    \
USE_CUDA_PATH=/usr/local/cuda \
USE_CUDNN=1                   \
-j\$(nproc)
"""
                        make('mklml_gpu', flag)
                        pack_lib('mklml')
                    }
                }
            },
            'CPU windows'{
                node('windows') {
                    ws('workspace/build-cpu') {
                    bat """rmdir /s/q build_vc14_cpu
mkdir build_vc14_cpu
cd build_vc14_cpu
cmake -G \"Visual Studio 14 2015 Win64\" -DUSE_CUDA=0 -DUSE_CUDNN=0 -DUSE_NVRTC=0 -DUSE_OPENCV=1 -DUSE_OPENMP=1 -DUSE_PROFILER=1 -DUSE_BLAS=open -DUSE_DIST_KVSTORE=0 ${env.WORKSPACE}/mxnet"""
                    }
                }
            },
            'GPU windows'{
                node('windows') {
                    ws('workspace/build-gpu') {
                        bat """rmdir /s/q build_vc14_gpu
mkdir build_vc14_gpu
call "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\bin\\x86_amd64\\vcvarsx86_amd64.bat"
cd build_vc14_gpu
cmake -G \"NMake Makefiles JOM\" -DUSE_CUDA=1 -DUSE_CUDNN=1 -DUSE_NVRTC=1 -DUSE_OPENCV=1 -DUSE_OPENMP=1 -DUSE_PROFILER=1 -DUSE_BLAS=open -DUSE_DIST_KVSTORE=0 -DCUDA_ARCH_NAME=All -DCMAKE_CXX_FLAGS_RELEASE=/FS -DCMAKE_BUILD_TYPE=Release ${env.WORKSPACE}/mxnet"""
                    }
                }
            }
}

// Python unittest for CPU
def python_ut(docker_type) {
    timeout(time: max_time, unit: 'MINUTES') {
        sh "${docker_run} ${docker_type} PYTHONPATH=./python/ nosetests --with-timer --verbose tests/python/unittest"
        sh "${docker_run} ${docker_type} PYTHONPATH=./python/ nosetests-3.4 --with-timer --verbose tests/python/unittest"
    }
}

// GPU test has two parts. 1) run unittest on GPU, 2) compare the results on
// both CPU and GPU
def python_gpu_ut(docker_type) {
    timeout(time: max_time, unit: 'MINUTES') {
        sh "${docker_run} ${docker_type} PYTHONPATH=./python/ nosetests --with-timer --verbose tests/python/gpu"
        sh "${docker_run} ${docker_type} PYTHONPATH=./python/ nosetests-3.4 --with-timer --verbose tests/python/gpu"
    }
}

stage('Unit Test') {
    parallel 'Python2/3: CPU': {
        node('linux') {
            ws('workspace/ut-python-cpu') {
                init_git()
                unpack_lib('cpu')
                python_ut('cpu')
            }
        }
    },
            'Python2/3: GPU': {
                node('GPU' && 'linux') {
                    ws('workspace/ut-python-gpu') {
                        init_git()
                        unpack_lib('gpu', mx_lib)
                        python_gpu_ut('gpu')
                    }
                }
            },
            'Python2/3: MKLML': {
                node('linux') {
                    ws('workspace/ut-python-mklml') {
                        init_git()
                        unpack_lib('mklml')
                        python_ut('mklml_gpu')
                        python_gpu_ut('mklml_gpu')
                    }
                }
            },
            'Scala: CPU': {
                node('linux') {
                    ws('workspace/ut-scala-cpu') {
                        init_git()
                        unpack_lib('cpu')
                        timeout(time: max_time, unit: 'MINUTES') {
                            sh "${docker_run} cpu make scalapkg USE_BLAS=openblas"
                            sh "${docker_run} cpu make scalatest USE_BLAS=openblas"
                        }
                    }
                }
            }
}


stage('Integration Test') {
    parallel 'Python': {
        node('GPU' && 'linux') {
            ws('workspace/it-python-gpu') {
                init_git()
                unpack_lib('gpu')
                timeout(time: max_time, unit: 'MINUTES') {
                    sh "${docker_run} gpu PYTHONPATH=./python/ python example/image-classification/test_score.py"
                }
            }
        }
    },
            'Caffe': {
                node('GPU' && 'linux') {
                    ws('workspace/it-caffe') {
                        init_git()
                        unpack_lib('gpu')
                        timeout(time: max_time, unit: 'MINUTES') {
                            sh "${docker_run} caffe_gpu PYTHONPATH=/caffe/python:./python python tools/caffe_converter/test_converter.py"
                        }
                    }
                }
            }
}
