#include "ThreadPool.h"

ThreadPool::ThreadPool(size_t numThreads)
{
	for (int i = 0; i < numThreads; i++)
	{
        worker.emplace_back([this] {
            while (true) {
                std::function<void()> task;

                {
                    std::unique_lock<std::mutex> lock(queueMutex);

                    cv.wait(lock, [this] {
                        return stopThreads || !tasks.empty();
                    });

                    if (stopThreads && tasks.empty())
                        return;

                    task = std::move(tasks.front());
                    tasks.pop();
                }

                task();
            }
        });
	}
}

ThreadPool::~ThreadPool()
{
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        stopThreads = true;
    }

    cv.notify_all();

    for (auto& t : worker)
        t.join();
}
