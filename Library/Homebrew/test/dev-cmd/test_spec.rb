require "cmd/shared_examples/args_parse"

describe "Homebrew.test_args" do
  it_behaves_like "parseable arguments"
end

describe "brew test", :integration_test do
  it "tests a given Formula" do
    install_test_formula "testball", <<~'RUBY'
      test do
        assert_equal "test", shell_output("#{bin}/test")
      end
    RUBY

    expect { brew "test", "--verbose", "testball" }
      .to output(/Testing testball/).to_stdout
      .and not_to_output.to_stderr
      .and be_a_success
  end
end
