# frozen_string_literal: true

shared_examples "formulae exist" do |array|
  array.each do |f|
    it "#{f} formula exists" do
      formula_path = Pathname("#{HOMEBREW_LIBRARY_PATH}/../Taps/homebrew/homebrew-core/Formula/#{f}.rb")
      expect(formula_path.exist?).to be true
    end
  end
end
