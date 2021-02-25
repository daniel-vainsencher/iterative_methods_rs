# prereq: sudo npm install -g browser-sync
browser-sync start --ss target/doc -s target/doc --directory --no-open --no-inject-changes --watch &
cargo watch -x check -x fmt -x doc -x clippy -x run --examples -x test 
