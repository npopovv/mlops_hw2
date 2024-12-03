# mlops_hw1

* Реализованы возможности обучения и предикта 2х моделей: логистическая регрессия и дерево. Также есть возможность узнать, какие модели уже были обучены, удалить какую-то, узнать статус сервера.

* Для запуска сервиса на FASTAPI, нужно выполнить команду: uvicorn app.fastapi.main:app --reload. Протестировать можно с помощью swagger UI, перейдя по ссылке: http://127.0.0.1:8000/docs.

* Также протестировать можно с помощью дашборда, запустив команду streamlit run app/fastapi/dash_fastapi.py

* Такой же сервер реализован с помощью gRPC. Для запуска сервера запускаем команду python app/grpc/grpc_server.py. Также для тестирования реализован клиент, запускается так: python app/grpc/grpc_server.py.

UPD: исправлены ошибки
1) Код разделен на директории-подмодули.
2) В gitignore добавлены ненужные файлы.
3) Добавлен отдельный файл с конфигом логгирования.
4) Убран индекс моделей. Теперь инфа о моделях достается из папки models_storage, в которой хранятся пиклы. Позволяет также знать о обученных при прошлом запуске сервера моделях.  
5) **Добавлена подача данных при обучении модели.**
6) В статусе сервера fastApi теперь также отображается процент занятой оперативки, время работы сервера, все ли ок.  
7) **Проведена проверка с помощью black и ruff. Сгенерированные в grpc файлы помещены в exclude в pyproject.toml.**

Команда: Попов Никита, Фролов Кирилл, Изотова Анастасия.
