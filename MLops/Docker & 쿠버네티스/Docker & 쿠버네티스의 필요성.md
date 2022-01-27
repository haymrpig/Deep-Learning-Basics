- **데이터 수집**

  Sqoop, Flume, Kafka, Flink, Spark Streaming, Airflow

- **데이터 저장**

  MySQL, Hadoop, Amazon S3, MiniO

- **데이터 관리**

  TFDV, DVC, Feast, Amundsen

  

- **모델 개발**

  Jupyter Hub, Docker, Kubeflow, Optuna, Ray, katib

- **모델 버전 관리**

  Git, MLflow, Github Action, Jenkins

- **모델 학습 스케쥴링 관리**

  Grafana, Kubernetes

$$
X\cdot Y=(x_{ij}*y_{ij})
$$



- **모델 패키징**

  Docker, Flask, FastAPI, BentoML, Kubeflow, TFServing, seldon-core

- **서빙 모니터링**

  Prometheus, Grafana, Thanos

- **파이프라인 매니징**

  Kubeflow, argo workflows, Airflow



# Docker & 쿠버네티스의 필요성

`Reproducibility` : 실행 환경의 일관성 & 독립성 보장

`Job Scheduling` : 스케줄 관리, 병렬 작업 관리, 유효 자원 관리

`Auto-healing & Auto-scaling` : 장애 대응, 트래픽 대응

### Docker

- **Container Orchestration**

  컨테이너 오케스트레이션은 컨테이너의 배포, 관리, 확장, 네트워킹을 자동화하는 기술이다. 

  1. 쿠버네티스와 같은 컨테이너 오케스트레이션 툴을 사용할 때는 YAML 또는 JSON파일을 사용해 애플리케이션 설정에 대해 정의하고, 설정 파일은 설정 관리 툴에 컨테이너 이미지의 위치 및 네트워크 구축법, 로그 저장 장소를 포함한다. 

  2. 새 컨테이너를 배포할 때 컨테이너 관리 도구는 정의된 요구사항을 고려하여 배포를 클러스터에 자동으로 예약하고 적당한 호스트를 찾는다. 이후, 오케스트레이션 툴이 작성 파일에 정의된 사양에 따라 컨테이너 라이프사이클을 관리한다. 

  쉽게 말해서, 컨테이너의 특성에 따라 어떤 서버에 배치할지를 자동으로 결정해주는 것을 예로 들 수 있다. GPU를 많이 사용하는 컨테이너의 경우, GPU가 많은 서버에, 메모리를 많이 사용하는 컨테이너의 경우 메모리가 많은 서버에 배치해주는 것 등이 있다. 

  위 예시 외에도, 여러 사용자가 함께 서버를 공유하며 각자의 모델 학습을 돌리려 할때, GPU자원이 남았는지, 학습 후 서버 자원을 정리하거나 이런 귀찮은 일 없이 자동으로 해주는 것을 의미한다고 생각하면 된다.  

- **Docker의 개념**

  애플리케이션을 어떤 os, 어떤 환경에서도 동일하게 실행하게 해주는 컨테이너화 프로그램이다. 

