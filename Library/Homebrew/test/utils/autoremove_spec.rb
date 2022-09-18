# typed: false
# frozen_string_literal: true

require "utils/autoremove"

describe Utils::Autoremove do
  shared_context "with formulae for dependency testing" do
    let(:formula_with_deps) do
      formula "zero" do
        url "zero-1.0"

        depends_on "three" => :build
      end
    end

    let(:formula_is_dep1) do
      formula "one" do
        url "one-1.1"
      end
    end

    let(:formula_is_dep2) do
      formula "two" do
        url "two-1.1"
      end
    end

    let(:formula_is_build_dep) do
      formula "three" do
        url "three-1.1"
      end
    end

    let(:formulae) do
      [
        formula_with_deps,
        formula_is_dep1,
        formula_is_dep2,
        formula_is_build_dep,
      ]
    end

    let(:tab_from_keg) { double }

    before do
      allow(formula_with_deps).to receive(:runtime_formula_dependencies).and_return([formula_is_dep1,
                                                                                     formula_is_dep2])
      allow(formula_is_dep1).to receive(:runtime_formula_dependencies).and_return([formula_is_dep2])

      allow(Tab).to receive(:for_keg).and_return(tab_from_keg)
    end
  end

  describe "::formulae_with_no_formula_dependents" do
    include_context "with formulae for dependency testing"

    before do
      allow(Formulary).to receive(:factory).with("three").and_return(formula_is_build_dep)
    end

    context "when formulae are bottles" do
      it "filters out runtime dependencies" do
        allow(tab_from_keg).to receive(:poured_from_bottle).and_return(true)
        expect(described_class.send(:formulae_with_no_formula_dependents, formulae))
            .to eq([formula_with_deps, formula_is_build_dep])
      end
    end

    context "when formulae are built from source" do
      it "filters out runtime and build dependencies" do
        allow(tab_from_keg).to receive(:poured_from_bottle).and_return(false)
        expect(described_class.send(:formulae_with_no_formula_dependents, formulae))
            .to eq([formula_with_deps])
      end
    end
  end

  describe "::unused_formulae_with_no_formula_dependents" do
    include_context "with formulae for dependency testing"

    before do
      allow(tab_from_keg).to receive(:poured_from_bottle).and_return(true)
    end

    specify "installed on request" do
      allow(tab_from_keg).to receive(:installed_on_request).and_return(true)
      expect(described_class.send(:unused_formulae_with_no_formula_dependents, formulae))
          .to eq([])
    end

    specify "not installed on request" do
      allow(tab_from_keg).to receive(:installed_on_request).and_return(false)
      expect(described_class.send(:unused_formulae_with_no_formula_dependents, formulae))
          .to match_array(formulae)
    end
  end

  shared_context "with formulae and casks for dependency testing" do
    include_context "with formulae for dependency testing"

    require "cask/cask_loader"

    let(:cask_one_dep) do
      Cask::CaskLoader.load(+<<-RUBY)
        cask "red" do
          depends_on formula: "two"
        end
      RUBY
    end

    let(:cask_multiple_deps) do
      Cask::CaskLoader.load(+<<-RUBY)
        cask "blue" do
          depends_on formula: "zero"
        end
      RUBY
    end

    let(:cask_no_deps1) do
      Cask::CaskLoader.load(+<<-RUBY)
        cask "green" do
        end
      RUBY
    end

    let(:cask_no_deps2) do
      Cask::CaskLoader.load(+<<-RUBY)
        cask "purple" do
        end
      RUBY
    end

    let(:casks_no_deps) { [cask_no_deps1, cask_no_deps2] }
    let(:casks_one_dep) { [cask_no_deps1, cask_no_deps2, cask_one_dep] }
    let(:casks_multiple_deps) { [cask_no_deps1, cask_no_deps2, cask_multiple_deps] }

    before do
      allow(Formula).to receive("[]").with("zero").and_return(formula_with_deps)
      allow(Formula).to receive("[]").with("one").and_return(formula_is_dep1)
      allow(Formula).to receive("[]").with("two").and_return(formula_is_dep2)
    end
  end

  describe "::formulae_with_cask_dependents" do
    include_context "with formulae and casks for dependency testing"

    specify "no dependents" do
      expect(described_class.send(:formulae_with_cask_dependents, casks_no_deps))
        .to eq([])
    end

    specify "one dependent" do
      expect(described_class.send(:formulae_with_cask_dependents, casks_one_dep))
        .to eq([formula_is_dep2])
    end

    specify "multiple dependents" do
      expect(described_class.send(:formulae_with_cask_dependents, casks_multiple_deps))
        .to match_array([formula_with_deps, formula_is_dep1, formula_is_dep2])
    end
  end
end
