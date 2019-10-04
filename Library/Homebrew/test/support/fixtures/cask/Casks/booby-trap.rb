cask 'booby-trap' do
  version '0.0.7'

  url do
    # to be lazily evaluated
    raise 'Boom'
  end
end
