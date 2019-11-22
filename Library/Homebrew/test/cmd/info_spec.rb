# frozen_string_literal: true

require "cmd/info"

require "cmd/shared_examples/args_parse"

describe "Homebrew.info_args" do
  it_behaves_like "parseable arguments"
end

describe "brew info", :integration_test do
  it "prints as json with the --json=v1 flag" do
    setup_test_formula "testball"

    expect { brew "info", "testball", "--json=v1" }
      .to output(a_json_string).to_stdout
      .and not_to_output.to_stderr
      .and be_a_success
  end
end

describe Homebrew do
  let(:remote) { "https://github.com/Homebrew/homebrew-core" }

  specify "::github_remote_path" do
    expect(subject.github_remote_path(remote, "Formula/git.rb"))
      .to eq("https://github.com/Homebrew/homebrew-core/blob/master/Formula/git.rb")

    expect(subject.github_remote_path("#{remote}.git", "Formula/git.rb"))
      .to eq("https://github.com/Homebrew/homebrew-core/blob/master/Formula/git.rb")

    expect(subject.github_remote_path("git@github.com:user/repo", "foo.rb"))
      .to eq("https://github.com/user/repo/blob/master/foo.rb")

    expect(subject.github_remote_path("https://mywebsite.com", "foo/bar.rb"))
      .to eq("https://mywebsite.com/foo/bar.rb")
  end
end
