# typed: false
# frozen_string_literal: true

require "cmd/shared_examples/args_parse"

describe "Homebrew.cleanup_args" do
  it_behaves_like "parseable arguments"
end

describe "brew cleanup", :integration_test do
  before do
    FileUtils.mkdir_p HOMEBREW_LIBRARY/"Homebrew/vendor/"
    FileUtils.touch HOMEBREW_LIBRARY/"Homebrew/vendor/portable-ruby-version"
  end

  after do
    FileUtils.rm_rf HOMEBREW_LIBRARY/"Homebrew"
  end

  describe "--prune=all" do
    it "removes all files in Homebrew's cache" do
      (HOMEBREW_CACHE/"test").write "test"

      expect { brew "cleanup", "--prune=all" }
        .to output(%r{#{Regexp.escape(HOMEBREW_CACHE)}/test}o).to_stdout
        .and not_to_output.to_stderr
        .and be_a_success
    end
  end
end
