# frozen_string_literal: true

require "cmd/shared_examples/args_parse"

describe "Homebrew.pull_args" do
  it_behaves_like "parseable arguments"
end

describe "brew pull", :integration_test do
  it "fetches a patch from a GitHub commit or pull request and applies it", :needs_network do
    HOMEBREW_REPOSITORY.cd do
      system "git", "init"
      system "git", "remote", "add", "origin", "https://github.com/Homebrew/brew"
    end

    expect { brew "pull", "https://github.com/Homebrew/brew/pull/1249" }
      .to output(/Fetching patch/).to_stdout
      .and be_a_failure
  end
end
