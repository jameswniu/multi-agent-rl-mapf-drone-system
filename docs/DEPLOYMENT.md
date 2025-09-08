\# Deployment Guide



1\. Build image

&nbsp;  ```bash

&nbsp;  make build


2. Run locally



bash

Copy code

make run


3. Deploy to Kubernetes



kubectl apply -f docker/k8s/deployment.yaml

kubectl apply -f docker/k8s/service.yaml

yaml



---



\# monitoring



\*\*`monitoring/alertmanager.yml`\*\*

```yaml

\# Alertmanager routes Prometheus alerts to channels (e.g. Slack)



route:

&nbsp; receiver: 'slack-notifications'



receivers:

\- name: 'slack-notifications'

&nbsp; slack\_configs:

&nbsp; - channel: '#alerts'

&nbsp;   send\_resolved: true

&nbsp;   text: "ALERT: {{ .CommonAnnotations.summary }}"

