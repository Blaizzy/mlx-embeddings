# frozen_string_literal: true

require "cmd/shared_examples/args_parse"

describe "Homebrew.pull_args" do
  it_behaves_like "parseable arguments"
end

describe "brew pull", :integration_test do
  it "fetches a patch from a GitHub commit or pull request and applies it", :needs_network do
    CoreTap.instance.path.cd do
      system "git", "init"
      system "git", "checkout", "-b", "new-branch"
    end

    expect { brew "pull", "https://github.com/Homebrew/brew/pull/1249" }
      .to output(/Fetching patch/).to_stdout
      .and output(/Patch failed to apply/).to_stderr
      .and be_a_failure
  end
end
