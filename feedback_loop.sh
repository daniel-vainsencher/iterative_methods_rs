# prereq: sudo npm install -g browser-sync
browser-sync start --ss target/doc/iterative_methods -s target/doc/iterative_methods --directory --no-open --no-inject-changes --watch &
cargo watch -x check -x fmt -x doc -x clippy -x build -x test
