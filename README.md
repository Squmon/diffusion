# Общая инфа
Генерации лежат в generations, а дполнительные изображения в папке old
## Что делал
1) Обучены 31 автонкодеров, выбран самый сильный - лежит в папке моделс.
конфиг и веса лежат в соответствующей папке
2) Обучена flow matching модель для генерации сэмплов из латентного представления VAE.

Так же нужно добавить что для более стабильного flow matching я нормировал латентное пространство котов, что бы среднее было в 0, а стандартное отклонение = 1

## Зависимости
keras, tensorflow, numpy
# Интересные наблюдения
1) Автоэнкодер был обучен на маленьком и специфичном датасете из пиксель артов, но не смотря на это неплохо кодирует любое изображение. Скорее всего, модель научилась кодировать и декодировать текстуры, но не сами объекты.
2) В данном проекте, увиличение пространственного разрешения латентного пространства лучше сказывалось на качестве восстановленных изображений, чем увиличение каналов

# Как пользоватся

Основная масса кода для ВАЕ, т.к я угарнул или был под веществами и написал черезчур слодный пайплайн. Там его обучение, но вы главное импортируйте то что нужно

## Получение

Загрузка VAE
```python
import json
from models import get_vae_from_config

with open("config.json", 'r') as j:
    config = json.load(j)

vae = get_vae_from_config(config)
vae.load_weights("StrongSon.weights.h5")
time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
latent_shape = vae.decoder.input_shape[1:]
```

Получение выходов энкодера - z_mean, z_log_var и выходов декодера - reconstruction
```python
reconstruction, z_mean, z_log_var = vae.call(original_image)
```
или
```python
vae.encoder(original_image)
vae.decoder(sample_from_latent) # на вход ожидает батчи изображений.
```

Получение Flow matching модели куда проще
```python
unet = keras.models.load_model("small_flow_cats.keras")
```
Использование Flow matching
```python
def sample(model, noise, num_steps=50):
    dt = 1.0 / num_steps
    x = noise
    for i in range(num_steps):
        t = tf.cast(i / num_steps, tf.float32)
        t_input = tf.fill((noise.shape[0], 1), t)
        v = model({"x": x, "t": t_input}, training=False)
        x = x + v * dt

    return x

from stdMeans import cats # это вопросы нормировки, главное получить std и mean, что бы пространства совпадали
std, mean = cats()

noise = tf.random.normal(shape=(num_samples, *latent_shape))
x = sample(unet, noise, num_steps = 100)*std + mean
samples = (vae.decoder(x) + 1.0)/2.0
```
Если вам совсем лень, то есть готовый скрипт
---
генерит 4 анимации генерации (как оно из шума делает картинку):
`python sampling.py --num 4 --steps 50 --trajectory True --dest generations` 

---
`python sampling.py --num 20 --steps 50 --dest generations` генерит 20 картинок
Чем больше параметр steps, тем лучше будет итоговое изображение (но после 50 увеличивать смысла нет)