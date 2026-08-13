//! Helper for running async futures synchronously across any runtime environment.

use std::future::Future;

/// Run a future to completion synchronously.
///
/// Handles all runtime environments safely:
/// - If running inside a multi-threaded Tokio runtime, offloads execution to a worker thread
///   via `std::thread::spawn` wrapped in `tokio::task::block_in_place`.
/// - If running without a Tokio runtime, builds a temporary single-threaded runtime.
pub fn block_on_future<F>(fut: F) -> F::Output
where
    F: Future + Send,
    F::Output: Send,
{
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => match handle.runtime_flavor() {
            tokio::runtime::RuntimeFlavor::CurrentThread => std::thread::scope(|s| {
                s.spawn(move || {
                    tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .expect("Failed to build current_thread runtime")
                        .block_on(fut)
                })
                .join()
                .expect("Blocking task thread panicked")
            }),
            _ => std::thread::scope(|s| {
                tokio::task::block_in_place(|| {
                    s.spawn(move || handle.block_on(fut))
                        .join()
                        .expect("Blocking task thread panicked")
                })
            }),
        },
        Err(_) => {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to create temporary tokio runtime");
            rt.block_on(fut)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_on_future_no_runtime() {
        let res = block_on_future(async { 42 });
        assert_eq!(res, 42);
    }

    #[tokio::test]
    async fn test_block_on_future_inside_tokio() {
        let res = block_on_future(async { "hello" });
        assert_eq!(res, "hello");
    }
}
