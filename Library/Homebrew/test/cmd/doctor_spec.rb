# frozen_string_literal: true

require "cmd/shared_examples/args_parse"

describe "Homebrew.doctor_args" do
  it_behaves_like "parseable arguments"
end

describe "brew doctor", :integration_test do
  specify "check_integration_test" do
    expect { brew "doctor", "check_integration_test" }
      .to output(/This is an integration test/).to_stderr
  end
end
