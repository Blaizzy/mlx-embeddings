# typed: false
# frozen_string_literal: true

require "cmd/shared_examples/args_parse"

describe "Homebrew.completions_args" do
  it_behaves_like "parseable arguments"
end

describe "brew completions", :integration_test do
  it "runs the status subcommand correctly" do
    HOMEBREW_REPOSITORY.cd do
      system "git", "init"
    end

    brew "completions", "link"
    expect { brew "completions" }
      .to output(/Completions are linked/).to_stdout
      .and not_to_output.to_stderr
      .and be_a_success

    brew "completions", "unlink"
    expect { brew "completions" }
    .to output(/Completions are not linked/).to_stdout
    .and not_to_output.to_stderr
    .and be_a_success
  end
end
