# frozen_string_literal: true

require "cmd/shared_examples/args_parse"

describe "Homebrew.__env_args" do
  it_behaves_like "parseable arguments"
end

describe "brew --env", :integration_test do
  describe "--shell=bash" do
    it "prints the Homebrew build environment variables in Bash syntax" do
      expect { brew "--env", "--shell=bash" }
        .to output(/export CMAKE_PREFIX_PATH="#{Regexp.quote(HOMEBREW_PREFIX)}"/).to_stdout
        .and not_to_output.to_stderr
        .and be_a_success
    end
  end
end
