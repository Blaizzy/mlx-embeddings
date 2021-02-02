# typed: false
# frozen_string_literal: true

require "cmd/shared_examples/args_parse"

describe "brew irb" do
  it_behaves_like "parseable arguments"

  it "starts an interactive Homebrew shell session", :integration_test do
    setup_test_formula "testball"

    irb_test = HOMEBREW_TEMP/"irb-test.rb"
    irb_test.write <<~RUBY
      "testball".f
      :testball.f
      exit
    RUBY

    expect { brew "irb", irb_test }
      .to output(/Interactive Homebrew Shell/).to_stdout
      .and not_to_output.to_stderr
      .and be_a_success
  end
end
