# frozen_string_literal: true

require "cmd/shared_examples/args_parse"

describe "Homebrew.sh_args" do
  it_behaves_like "parseable arguments"
end

describe "brew sh", :integration_test do
  it "runs a shell with the Homebrew environment" do
    expect { brew "sh", "SHELL" => which("true") }
      .to output(/Your shell has been configured/).to_stdout
      .and not_to_output.to_stderr
      .and be_a_success
  end
end
