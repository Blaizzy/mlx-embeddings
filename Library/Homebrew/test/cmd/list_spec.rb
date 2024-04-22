# frozen_string_literal: true

require "cmd/list"
require "cmd/shared_examples/args_parse"
require "tab"

RSpec.describe Homebrew::Cmd::List do
  def setup_installation(formula_name, installed_on_request:)
    setup_test_formula(formula_name)

    keg_dir = HOMEBREW_CELLAR/formula_name/"1.0"
    keg_dir.mkpath

    tab = Tab.new(
      "installed_on_request" => installed_on_request,
      "tabfile"              => keg_dir/Tab::FILENAME,
    )
    tab.write

    keg_dir
  end

  let(:formulae) { %w[bar foo qux] }

  it_behaves_like "parseable arguments"

  it "prints all installed Formulae", :integration_test do
    formulae.each do |f|
      (HOMEBREW_CELLAR/f/"1.0/somedir").mkpath
    end

    expect { brew "list", "--formula" }
      .to output("#{formulae.join("\n")}\n").to_stdout
      .and not_to_output.to_stderr
      .and be_a_success
  end

  it "lists the formulae installed on request or automatically",
     :integration_test do
    setup_installation "foo", installed_on_request: true
    setup_installation "bar", installed_on_request: false

    expect { brew "list", "--manual" }
      .to be_a_success
      .and output("foo\n").to_stdout
      .and not_to_output.to_stderr

    expect { brew "list", "--auto" }
      .to be_a_success
      .and output("bar\n").to_stdout
      .and not_to_output.to_stderr

    expect { brew "list", "--manual", "--auto" }
      .to be_a_success
      .and output("bar: auto\nfoo: manual\n").to_stdout
      .and not_to_output.to_stderr
  end
end
