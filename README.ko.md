# Anima Lightning for Forge Neo

Forge Neo 원본과 다른 확장의 파일을 수정하지 않는 로컬 확장입니다.
공식 [aina-tech/Anima-Lightning](https://huggingface.co/aina-tech/Anima-Lightning)의
shift 3 / TDM-Unify 배포본을 대상으로 합니다.

## 설치 및 모델 준비

Forge의 **Extensions → Install from URL**에서
`https://github.com/david419kr/sd-forge-anima-lightning`을 설치하고 재시작합니다.
모델 가중치는 저장소에 포함되어 있지 않습니다.
처음 설치한다면 [README의 다운로드·변환 명령](README.md#prepare-the-checkpoint)을 먼저 실행하세요.
기본 Anima가 동작하는 Forge Neo 환경과 해당 텍스트 인코더/VAE가 필요합니다.

## 사용

1. Forge Neo를 새로 실행합니다.
2. 체크포인트에서 `Anima-Lightning-K4-shift3-forge.safetensors`를 선택합니다.
3. 기존 Anima 구성요소인 `q306b.safetensors`, `qi.vae.safetensors`를 선택합니다.
4. 샘플러에서 `Anima Lightning`을 선택하고, 첫 실행은 4스텝·CFG 1·Shift 3으로 설정합니다.
5. 처음에는 1024×1024, Hires 없이 사용합니다. NegPiP은 기존 UI에서 켤 수 있습니다.

확장에는 프리셋 체크박스나 어코디언이 없으며, 사용자의 샘플러·스텝·CFG·Shift를 덮어쓰지 않습니다.
4스텝에서 전용 샘플러의 스케줄은 기존과 동일한 `[1, .9, .75, .5, 0]`입니다.
6·8스텝 등으로 바꾸면 스케줄 길이도 맞춰집니다. 전용 샘플러는 Euler와 **고정 shift 3**을
사용하므로 스케줄 타입과 UI Shift 선택은 이 샘플러의 sigma 배열을 바꾸지 않습니다.
다른 샘플러를 선택하면 그 샘플러의 스케줄 및 Shift 설정을 그대로 사용합니다.

학습된 K 분기는 선택한 샘플러와 관계없이 **UI의 스텝 수**에 연동됩니다.
기본 패스는 `Steps`, Hires는 `Hires steps`(0이면 기본 Steps)를 사용하며,
K 조건은 모델의 학습 설정대로 `3 / (K + 2)`로 계산합니다.
CFG는 사용자가 선택하며 1로 제한하지 않습니다. 4 이외 스텝 수에서 품질 향상을 보장하지 않습니다.
기존 파일명에 포함된 `K4`는 첫 변환 때의 이름일 뿐 현재 K를 제한하지 않으며 재변환할 필요가 없습니다.

Hires/img2img에서는 Forge가 denoise strength에 맞춰 내부 그리드를 확장·잘라 쓰는 동작을
유지합니다. 이때도 K는 해당 패스의 UI 스텝 수를 사용합니다. 이는 공식 txt2img 레시피를
응용한 경로이며 제작자가 검증한 별도 레시피는 아닙니다.

CFG 1에서는 일반 네거티브 프롬프트의 CFG 효과가 없습니다.
NegPiP의 양성 프롬프트 안 음수 가중치(`(blue hair:-1)`)는 별도 어텐션 경로로 처리됩니다.

## 호환 방식

- Forge의 엔진/모델 클래스와 원래 `forward`, 텍스트 엔진, 어텐션 함수를 유지합니다.
- 기존 로더를 감싸서 K 가중치가 있는 Anima 모델을 로드할 때만 추가 모듈을 구성합니다.
- 시간 임베딩과 정규화 모듈의 forward hook으로 학습된 K 분기를 더합니다.
- 추가 Linear/RMSNorm은 Forge의 dtype/오프로딩 연산 문맥 안에서 생성합니다.
- 매 샘플링 패스 전에 현재 모델의 K만 갱신하므로 다른 모델이나 이전 요청의 K가 섞이지 않습니다.
- 전용 샘플러는 기존 `KDiffusionSampler`와 CFG denoiser, 스크립트 콜백을 사용합니다.
- NegPiP/Forge Couple/LoRA 등이 사용하는 프롬프트·어텐션 경로를 우회하지 않습니다.
- 확장을 끄고 Forge를 재시작하면 원래 로더로 돌아갑니다.
- 예전 `Anima Lightning (4 steps)` 이름은 infotext/API 별칭으로 인식하되 목록에는 새 이름만 표시합니다.

설치된 확장을 동시에 로드할 수 있는 것과 모든 기능 조합의 결과가 검증된 것은 다릅니다.
Anima 3.8B의 Qwen3.5 어댑터 등 텍스트 인코더를 바꾸는 옵션은 공식 Lightning 조건과 달라집니다.
향후 Forge 내부 로더/시간 임베딩 구조가 바뀌면 확장 호환성 확인이 필요합니다.

## 모델 재변환

지원하는 원본 revision은 `9dbb6e1a0f94eda0a1e523836d32355fb961cc7a`이며,
다운로드 후 확장 내부의 `cache/source`에 저장합니다.
Transformer와 text conditioner만 사용하며, 다운로드된 Python 코드를 실행하지 않습니다.

```powershell
.\venv\Scripts\python.exe extensions\sd-forge-anima-lightning\tools\convert_model.py `
  extensions\sd-forge-anima-lightning\cache\source `
  models\Stable-diffusion\Anima-Lightning-K4-shift3-forge.safetensors
```

기존 출력 파일을 덮어쓰지 않습니다. 모든 원본 가중치를 보존하고 이름만 변환하며,
두 모듈에서 쓰는 학습된 K scale은 복제합니다. 알 수 없는 Transformer 키는 거부합니다.
원본 Transformer SHA-256:
`8ea644535ab333540320c6c3d13969c4487c1a6f1135898322317b0b0fb4bf99`.

모델 사용에는 원본 CircleStone Labs 비상업 라이선스와 upstream 조건이 적용됩니다.
원문 `LICENSE.md`, `NOTICE.md`는 `cache/source`에 있습니다.

## 검증

```powershell
.\venv\Scripts\python.exe extensions\sd-forge-anima-lightning\tests\test_support.py
```

Diffusers 시간 임베딩과 K=4/6/8 분기 수치 비교(FP32/BF16), 실제 NegPiP forward 래퍼 통과,
변환된 전체 텐서의 네이티브 모델/어댑터 키·shape 일치, 공식 스케줄을 검사합니다.
로컬 실행 검증 로그와 이미지는 `verification`에 저장하며 공개 저장소에는 포함하지 않습니다.

현재 설치 환경에서 실제 생성으로 확인한 항목:

| 경로 | 결과 |
|---|---|
| 1024×1024 Lightning 4스텝 | 통과 |
| 가변 K 변경 후 공식 4스텝 출력 | 이전 구현과 출력 픽셀 완전 일치 |
| 4→6→8→4스텝/K 전환 | 통과, 마지막 4스텝 출력도 동일 |
| 일반 Euler 6스텝·CFG 2·Karras·Shift 5 | 사용자 설정 유지, K=6 적용 |
| 프리셋 체크박스·어코디언 제거 | Gradio 구성과 API 인자에서 확인 |
| 이전 샘플러 이름으로 요청 | 새 `Anima Lightning`으로 연결 |
| NegPiP 끄기→켜기→끄기 | 켜기 전후 출력 픽셀 완전 일치, 켠 출력은 다름 |
| NegPiP + Forge Couple | 6스텝/K=6 통과 |
| NegPiP + ConDelta | 통과 |
| NegPiP + Anima LoRA + 단계 스케줄러 | 6스텝/K=6 통과 |
| NegPiP + Hires(512→768) | 기본 6스텝/K=6, Hires 8스텝/K=8·CFG 2 통과 |
| NegPiP + img2img(denoise 0.5) | 8스텝/K=8 통과 |
| 일반 Anima로 전환, Euler/2스텝/CFG 2 | K 분기 개입 없이 통과 |

ADetailer/ControlNet의 실제 실행, torch.compile, 교체 텍스트 인코더,
모든 확장의 모든 옵션 조합까지 검증한 것은 아닙니다.
검증에는 별도 설정 복사본을 사용했고 테스트 서버는 종료했습니다.
가변 K 변경의 상세 기록은 `verification/variable-k/report.json`에 있습니다.
이 기록 역시 개발에 사용한 로컬 설치에만 보관됩니다.

확장 소스 라이선스는 [AGPL v3](LICENSE)입니다. 모델 라이선스는 별도로 적용됩니다.
