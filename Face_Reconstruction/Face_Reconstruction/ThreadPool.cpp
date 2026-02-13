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

                task(); // Run job
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

template<typename F, typename... Args>
auto ThreadPool::ProcessThread(F&& f, Args&&... args)
->std::future<typename std::invoke_result<F, Args...>::type>
{
    using ReturnType =
        typename std::invoke_result<F, Args...>::type;

    auto task = std::make_shared<std::packaged_task<ReturnType()>>(
        std::bind(std::forward<F>(f),
            std::forward<Args>(args)...)
    );

    std::future<ReturnType> result = task->get_future();

    {
        std::lock_guard<std::mutex> lock(queueMutex);

        if (stopThreads)
            throw std::runtime_error("Submit on stopped ThreadPool");

        tasks.emplace([task]() { (*task)(); });
    }

    cv.notify_one();
    
    return result;
}