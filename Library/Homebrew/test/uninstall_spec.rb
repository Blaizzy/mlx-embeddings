# typed: false
# frozen_string_literal: true

require "uninstall"

describe Homebrew::Uninstall do
  let(:dependency) { formula("dependency") { url "f-1" } }
  let(:dependent) do
    formula("dependent") do
      url "f-1"
      depends_on "dependency"
    end
  end

  let(:kegs_by_rack) { { dependency.rack => [Keg.new(dependency.latest_installed_prefix)] } }

  before do
    [dependency, dependent].each do |f|
      f.latest_installed_prefix.mkpath
      Keg.new(f.latest_installed_prefix).optlink
    end

    tab = Tab.empty
    tab.homebrew_version = "1.1.6"
    tab.tabfile = dependent.latest_installed_prefix/Tab::FILENAME
    tab.runtime_dependencies = [
      { "full_name" => "dependency", "version" => "1" },
    ]
    tab.write

    stub_formula_loader dependency
    stub_formula_loader dependent
  end

  describe "::handle_unsatisfied_dependents" do
    specify "when developer" do
      ENV["HOMEBREW_DEVELOPER"] = "1"

      expect {
        described_class.handle_unsatisfied_dependents(kegs_by_rack)
      }.to output(/Warning/).to_stderr

      expect(Homebrew).not_to have_failed
    end

    specify "when not developer" do
      expect {
        described_class.handle_unsatisfied_dependents(kegs_by_rack)
      }.to output(/Error/).to_stderr

      expect(Homebrew).to have_failed
    end

    specify "when not developer and `ignore_dependencies` is true" do
      expect {
        described_class.handle_unsatisfied_dependents(kegs_by_rack, ignore_dependencies: true)
      }.not_to output.to_stderr

      expect(Homebrew).not_to have_failed
    end
  end
end
