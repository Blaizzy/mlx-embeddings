# typed: false
# frozen_string_literal: true

require "cmd/shared_examples/args_parse"
require "support/lib/config"

describe "Homebrew.home_args" do
  it_behaves_like "parseable arguments"
end

describe "brew home", :integration_test do
  let(:testballhome_homepage) {
    Formula["testballhome"].homepage
  }

  let(:local_caffeine_path) {
    cask_path("local-caffeine")
  }

  let(:local_caffeine_homepage) {
    Cask::CaskLoader.load(local_caffeine_path).homepage
  }

  it "opens the project page when no formula or cask is specified" do
    expect { brew "home", "HOMEBREW_BROWSER" => "echo" }
      .to output("https://brew.sh\n").to_stdout
      .and not_to_output.to_stderr
      .and be_a_success
  end

  it "opens the homepage for a given Formula" do
    setup_test_formula "testballhome"

    expect { brew "home", "testballhome", "HOMEBREW_BROWSER" => "echo" }
      .to output(/#{testballhome_homepage}/).to_stdout
      .and not_to_output.to_stderr
      .and be_a_success
  end

  it "opens the homepage for a given Cask" do
    expect { brew "home", local_caffeine_path, "HOMEBREW_BROWSER" => "echo" }
      .to output(/#{local_caffeine_homepage}/).to_stdout
      .and output(/Treating #{Regexp.escape(local_caffeine_path)} as a cask/).to_stderr
      .and be_a_success
    expect { brew "home", "--cask", local_caffeine_path, "HOMEBREW_BROWSER" => "echo" }
      .to output(/#{local_caffeine_homepage}/).to_stdout
      .and not_to_output.to_stderr
      .and be_a_success
  end

  it "opens the homepages for a given formula and Cask" do
    setup_test_formula "testballhome"

    expect { brew "home", "testballhome", local_caffeine_path, "HOMEBREW_BROWSER" => "echo" }
      .to output(/#{testballhome_homepage} #{local_caffeine_homepage}/).to_stdout
      .and output(/Treating #{Regexp.escape(local_caffeine_path)} as a cask/).to_stderr
      .and be_a_success
  end
end
