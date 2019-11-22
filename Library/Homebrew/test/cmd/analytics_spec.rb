# frozen_string_literal: true

require "cmd/shared_examples/args_parse"

describe "Homebrew.analytics_args" do
  it_behaves_like "parseable arguments"
end

describe "brew analytics", :integration_test do
  it "when HOMEBREW_NO_ANALYTICS is unset is disabled after running `brew analytics off`" do
    HOMEBREW_REPOSITORY.cd do
      system "git", "init"
    end

    brew "analytics", "off"
    expect { brew "analytics", "HOMEBREW_NO_ANALYTICS" => nil }
      .to output(/Analytics are disabled/).to_stdout
      .and not_to_output.to_stderr
      .and be_a_success
  end
end
