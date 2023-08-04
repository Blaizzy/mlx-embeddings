# frozen_string_literal: true

require "cmd/shared_examples/args_parse"

describe "brew test" do
  it_behaves_like "parseable arguments"

  it "tests a given Formula", :integration_test do
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

  describe "using inreplace" do
    it "replaces text in file", :integration_test do
      install_test_formula "testball", <<~RUBY
        test do
          (testpath/"file.txt").write "1"
          inreplace testpath/"file.txt" do |s|
            s.gsub! "1", "2"
          end
          assert_equal "2", (testpath/"file.txt").read
        end
      RUBY

      expect { brew "test", "--verbose", "testball" }
        .to output(/Testing testball/).to_stdout
        .and not_to_output.to_stderr
        .and be_a_success
    end

    it "fails when assertion fails", :integration_test do
      install_test_formula "testball", <<~RUBY
        test do
          (testpath/"file.txt").write "1"
          inreplace testpath/"file.txt" do |s|
            s.gsub! "1", "2"
          end
          assert_equal "3", (testpath/"file.txt").read
        end
      RUBY

      expect { brew "test", "--verbose", "testball" }
        .to output(/Testing testball/).to_stdout
        .and output(/Expected: "3"\n  Actual: "2"/).to_stderr
        .and be_a_failure
    end
  end
end
