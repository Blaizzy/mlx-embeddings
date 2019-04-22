# frozen_string_literal: true

require "cmd/shared_examples/args_parse"

describe "Homebrew.home_args" do
  it_behaves_like "parseable arguments"
end

describe "brew home", :integration_test do
  it "opens the homepage for a given Formula" do
    setup_test_formula "testballhome"

    expect { brew "home", "testballhome", "HOMEBREW_BROWSER" => "echo" }
      .to output("#{Formula["testballhome"].homepage}\n").to_stdout
      .and not_to_output.to_stderr
      .and be_a_success
  end
end
