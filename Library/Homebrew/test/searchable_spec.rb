# typed: false
# frozen_string_literal: true

require "searchable"

describe Searchable do
  subject(:searchable_collection) { collection.extend(described_class) }

  let(:collection) { ["with-dashes"] }

  describe "#search" do
    context "when given a block" do
      let(:collection) { [["with-dashes", "withdashes"]] }

      it "searches by the selected argument" do
        expect(searchable_collection.search(/withdashes/) { |_, short_name| short_name }).not_to be_empty
        expect(searchable_collection.search(/withdashes/) { |long_name, _| long_name }).to be_empty
      end
    end

    context "when given a regex" do
      it "does not simplify strings" do
        expect(searchable_collection.search(/with-dashes/)).to eq ["with-dashes"]
      end
    end

    context "when given a string" do
      it "simplifies both the query and searched strings" do
        expect(searchable_collection.search("with dashes")).to eq ["with-dashes"]
      end
    end

    context "when searching a Hash" do
      let(:collection) { { "foo" => "bar" } }

      it "returns a Hash" do
        expect(searchable_collection.search("foo")).to eq "foo" => "bar"
      end

      context "with a nil value" do
        let(:collection) { { "foo" => nil } }

        it "does not raise an error" do
          expect(searchable_collection.search("foo")).to eq "foo" => nil
        end
      end
    end
  end
end
